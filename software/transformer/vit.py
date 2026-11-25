"""Tiny ViT for CIFAR-10 whose attention can be softmax (teacher), linear (float) or hardware-quantised linear.

Each head has d_qk = 8 and d_v = 7, so that one head maps onto the 8x8 systolic array without tiling:
the 8th value slot carries the constant 1 that produces the linear-attention normaliser (see hw_attn.py).
Tokens: 4x4 patches of a 32x32 image -> N = 64 tokens (= the L0 / IFIFO depth of the core).

Quantisation (mode 'linear_q'):
  * QuantLinear for qkv / proj / fc1 / fc2: 4-bit signed weights and 4-bit signed inputs (LSQ, per tensor).
    The patch embedding and the classifier stay in float (first / last layer, as is common practice).
  * attention core per head: phiQ, phiK unsigned 4 bit, V signed 4 bit (LSQ scales per head), the state is
    requantised to 8 bit with a dynamic shared exponent, and the output is round(num / den) at V's scale.
    With `hw_exact = True` the core runs the bit-exact integer model in hw_attn.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_attn import D_QK, D_V, attn_core, bitlen

# ------------------------------------------------------------------------------------------- quantisers


def round_ste(x):
    return (x.round() - x).detach() + x


def grad_scale(s, g):
    return (s - s * g).detach() + s * g


class LSQ(nn.Module):
    """Learned-step-size quantiser. forward returns the INTEGER code (float tensor, STE gradients);
    multiply by .scale() to dequantise. `shape` gives per-channel scales (e.g. one per head)."""

    def __init__(self, lo, hi, shape=(1,)):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.s = nn.Parameter(torch.ones(shape))
        self.register_buffer('initialised', torch.tensor(0))

    def scale(self):
        return self.s.abs() + 1e-8

    def forward(self, x, view=None):
        if not self.initialised:
            with torch.no_grad():
                red = x.detach().abs()
                if self.s.numel() > 1:   # per head: x is [B, H, N, d]
                    m = red.transpose(0, 1).flatten(1).mean(1)
                else:
                    m = red.mean().view(1)
                self.s.data.copy_(2 * m / math.sqrt(self.hi) + 1e-4)
                self.initialised.fill_(1)
        s = self.scale()
        if view is not None:
            s = s.view(view)
        g = 1.0 / math.sqrt(x.numel() / max(1, self.s.numel()) * self.hi)
        s = grad_scale(s, g)
        return round_ste((x / s).clamp(self.lo, self.hi)), s


class QuantLinear(nn.Linear):
    """4-bit signed weight x 4-bit signed input, as executed on the array (integer GEMM, float rescale)."""

    def __init__(self, i, o, bias=True):
        super().__init__(i, o, bias)
        self.quant = False
        self.wq = LSQ(-8, 7)
        self.aq = LSQ(-8, 7)

    def forward(self, x):
        if not self.quant:
            return super().forward(x)
        xi, sx = self.aq(x)
        wi, sw = self.wq(self.weight)
        y = F.linear(xi, wi) * (sx * sw)
        return y + self.bias if self.bias is not None else y


# ------------------------------------------------------------------------------------------- model


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.h = heads
        self.q = QuantLinear(dim, heads * D_QK)
        self.k = QuantLinear(dim, heads * D_QK)
        self.v = QuantLinear(dim, heads * D_V)
        self.proj = QuantLinear(heads * D_V, dim)
        self.mode = 'softmax'
        self.hw_exact = False
        self.qq = LSQ(0, 15, (heads,))
        self.kq = LSQ(0, 15, (heads,))
        self.vq = LSQ(-8, 7, (heads,))
        self.capture = None   # dict -> integer tensors of the attention core are stored here (stimulus export)

    def core(self, x):
        """Attention output before the output projection: [B, H, N, D_V]."""
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, D_QK).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, D_QK).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, D_V).transpose(1, 2)
        if self.mode == 'softmax':
            a = (q @ k.transpose(-1, -2) * D_QK ** -0.5).softmax(-1)
            return a @ v
        qf, kf = F.relu(q), F.relu(k)
        if self.mode == 'linear':
            v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)
            P = qf @ (kf.transpose(-1, -2) @ v1)
            return P[..., :D_V] / (P[..., D_V:] + 1e-6)
        # 'linear_q': hardware-quantised core (float32, integer valued)
        with torch.autocast('cuda', enabled=False):
            hv = (1, -1, 1, 1)
            qi, _ = self.qq(qf.float(), hv)
            ki, _ = self.kq(kf.float(), hv)
            vi, sv = self.vq(v.float(), hv)
            if self.hw_exact:
                out, aux = attn_core(qi.detach().round(), ki.detach().round(), vi.detach().round())
                if self.capture is not None:
                    self.capture.update(phiq=qi.round().long(), phik=ki.round().long(), v=vi.round().long(),
                                        out=out, **aux)
                return out.float() * sv
            v1 = torch.cat([vi, torch.ones_like(vi[..., :1])], -1)
            S = ki.transpose(-1, -2) @ v1
            with torch.no_grad():
                sh = (bitlen(S.abs().flatten(-2).amax(-1)) - 7).clamp(min=0).float()[..., None, None]
            S8 = round_ste(S / 2 ** sh).clamp(-128, 127)
            P = qi @ S8
            den = P[..., D_V:]
            o = torch.where(den > 0, P[..., :D_V] / den.clamp(min=1e-6), torch.zeros_like(P[..., :D_V]))
            return round_ste(o).clamp(-8, 7) * sv

    def forward(self, x):
        o = self.core(x)
        B, H, N, _ = o.shape
        return self.proj(o.transpose(1, 2).reshape(B, N, H * D_V))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.fc1 = QuantLinear(dim, mlp)
        self.fc2 = QuantLinear(mlp, dim)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ViT(nn.Module):
    def __init__(self, dim=128, depth=6, heads=8, mlp=256, patch=4, n_cls=10):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, patch)
        n = (32 // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x).mean(1))

    def attns(self):
        return [b.attn for b in self.blocks]

    def set_mode(self, mode, hw_exact=False):
        for a in self.attns():
            a.mode, a.hw_exact = mode, hw_exact
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.quant = (mode == 'linear_q')

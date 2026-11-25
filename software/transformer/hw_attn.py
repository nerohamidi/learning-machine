"""Bit-exact integer model of the hardware linear-attention core (one head), shared by the PyTorch model
(hw-exact evaluation), the stimulus generator and the analysis scripts.

Hardware mapping (8x8 array, head with d_qk = 8 and d_v = 7):

  phase A (output stationary):  S^T = [V | 1]^T  phi(K)        S^T[r][c] = sum_n V1[n, r] * phiK[n, c]
        PE row r    = value index r (r = 7 is the constant-1 "value": it accumulates the normaliser z = sum_n phiK[n])
        PE column c = key feature c
        west  (L0)  : V1[n, :]    signed 4-bit activations       (sgn = 1)
        north (IFIFO): phiK[n, :] unsigned 4-bit weights          (weight-unsigned mode)
        drain       : one OFIFO vector per PE row r = S[:, r] = column r of S, exactly the word the WS kernel
                      load needs for PE column r in phase B, so no transpose is needed.
  requantiser (SFU): S (16 bit) -> S8 = clamp(round(S / 2^sh), -128, 127), sh = max(0, bitlen(OR_i |S_i|) - 7)
        (dynamic block floating point: one shared exponent per head, found with an OR-reduction, no calibration)
        S8 is split into nibbles  S8 = 16 * hi + lo,  hi = S8 >> 4 (signed), lo = S8 & 15 (unsigned)
  phase B (weight stationary): P = phiQ S8 = 16 * (phiQ hi) + (phiQ lo)   two passes, phiQ unsigned (sgn = 0),
        hi pass: weight signed, OFIFO << 4 -> pmem;  lo pass: weight unsigned, OFIFO + pmem -> divider
  divider (SFU): out[n, c] = round(P[n, c] / P[n, 7]) for c < 7, a signed 4-bit value at V's scale.
        Because phi >= 0, the attention weights are non-negative and sum to 1, so the output is a convex
        combination of V rows and fits V's own 4-bit range: no output scale, only a 4-quotient-bit divider.

All functions take integer-valued tensors (any float/int dtype) and return int64 tensors.
"""

import torch

D_QK = 8
D_V = 7
S8_MAX = 127
S8_MIN = -128


def bitlen(m):
    """Number of bits of a non-negative integer tensor (0 -> 0)."""
    m = m.to(torch.float64)
    _, e = torch.frexp(m)
    return torch.where(m > 0, e.to(torch.int64), torch.zeros_like(e, dtype=torch.int64))


def state(phik, v):
    """S[..., c, r] = sum_n phiK[n, c] * V1[n, r]   (c: key feature, r: value index incl. the constant 1)."""
    phik = phik.to(torch.float64)      # float64 matmul is exact here (|S| < 2^15) and also works on CUDA
    v1 = torch.cat([v.to(torch.float64), torch.ones_like(v[..., :1], dtype=torch.float64)], dim=-1)
    return (phik.transpose(-1, -2) @ v1).round().to(torch.int64)


def requant_shift(S):
    """Shared exponent per head: OR of |S| over the 8x8 block, keep 7 magnitude bits."""
    Sa = S.abs()
    m = Sa.flatten(-2).amax(-1)            # bitlen(max |S|) == bitlen(OR |S|)
    return (bitlen(m) - 7).clamp(min=0)


def requant(S, sh=None):
    """S8 = clamp(floor((S + 2^(sh-1)) / 2^sh), -128, 127)  (add-half then arithmetic shift)."""
    if sh is None:
        sh = requant_shift(S)
    sh_ = sh[..., None, None]
    half = torch.where(sh_ > 0, torch.ones_like(sh_) << (sh_ - 1).clamp(min=0), torch.zeros_like(sh_))
    S8 = torch.div(S + half, torch.ones_like(sh_) << sh_, rounding_mode='floor')
    return S8.clamp(S8_MIN, S8_MAX), sh


def nibbles(S8):
    hi = torch.div(S8, 16, rounding_mode='floor')   # arithmetic >> 4, signed -8..7
    lo = S8 - 16 * hi                                # 0..15
    return hi, lo


def hwdiv(n, d):
    """round(n / d), half away from zero, saturated to signed 4 bit; d == 0 -> 0.
    Hardware: t = 2|n| + d, 4 restoring compare-subtract stages against (2d << b), b = 3..0."""
    n = n.to(torch.int64)
    d = d.to(torch.int64)
    an = n.abs()
    dd = torch.where(d > 0, d, torch.ones_like(d))
    q = torch.div(2 * an + dd, 2 * dd, rounding_mode='floor').clamp(max=15)
    q = torch.where(n < 0, -q.clamp(max=8), q.clamp(max=7))
    return torch.where(d > 0, q, torch.zeros_like(q))


def attn_core(phiq, phik, v):
    """phiq, phik: [..., N, 8] in 0..15; v: [..., N, 7] in -8..7. Returns (out [..., N, 7] int4, aux dict)."""
    S = state(phik, v)
    S8, sh = requant(S)
    P = (phiq.to(torch.float64) @ S8.to(torch.float64)).round().to(torch.int64)   # == 16*(phiQ hi) + (phiQ lo)
    out = hwdiv(P[..., :D_V], P[..., D_V:D_V + 1])
    return out, dict(S=S, S8=S8, sh=sh, P=P)

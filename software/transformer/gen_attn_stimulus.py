"""+alpha (linear attention): stimulus / expected files for the hardware linear-attention core, one layer, all heads.

The integer operands are captured from the 4-bit ViT running its bit-exact integer attention core
(vit.py, hw_exact=True). As in Part 1, the hardware result is checked against the model itself:
out_int * s_v must equal the pre-hooked input of the output projection (attn.proj) of that head.

Files in --out (per head h):
  config.txt         n_heads N
  h{h}_v1.txt        N x 32b   west operand of phase A: lane r = V[n, r] (r < 7), lane 7 = 1   (signed 4b)
  h{h}_phik.txt      N x 32b   north operand of phase A: lane c = phiK[n, c]                    (unsigned 4b)
  h{h}_phiq.txt      N x 32b   west operand of phase B: lane c = phiQ[n, c]                     (unsigned 4b)
  h{h}_S.txt         8 x 128b  line r = S[:, r] (16b lanes c): the drained state, before requantisation
  h{h}_s8hi.txt      8 x 32b   line t = high nibbles of S8[:, t]: the phase-B PE weight word for column t
  h{h}_s8lo.txt      8 x 32b   line t = low nibbles
  h{h}_out.txt       N x 128b  lane r < 7: out[n, r] (signed 4b, sign-extended to 16b), lane 7: den = P[n, 7]
  stats.txt          per-head shift, zero fractions

Usage (from project/software/transformer, env ml):
    python gen_attn_stimulus.py --layer 0 --image 0 --out ../../hardware/sim/stim_attn_l0
    python gen_attn_stimulus.py --random --out ../../hardware/sim/stim_attn_rand
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hw_attn import attn_core, nibbles, D_V  # noqa: E402


def pack(vals, bw):
    return ''.join(format(int(v) & ((1 << bw) - 1), '0{}b'.format(bw)) for v in reversed(list(vals)))


def write(path, header, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(header + lines) + '\n')


def export(out_dir, phiq, phik, v, tag=''):
    """phiq, phik: [H, N, 8]; v: [H, N, 7] integer tensors."""
    os.makedirs(out_dir, exist_ok=True)
    H, N, _ = phiq.shape
    out, aux = attn_core(phiq, phik, v)
    S, S8, sh, P = aux['S'], aux['S8'], aux['sh'], aux['P']
    hi, lo = nibbles(S8)
    assert S.abs().max() < 2 ** 15 and P.abs().max() < 2 ** 15
    stats = []
    for h in range(H):
        v1 = torch.cat([v[h], torch.ones(N, 1, dtype=torch.long)], 1)
        write(os.path.join(out_dir, f'h{h}_v1.txt'), ['#lane7=1#', '#lane_r=V[n,r]#', '#signed4#'],
              [pack(v1[n].tolist(), 4) for n in range(N)])
        write(os.path.join(out_dir, f'h{h}_phik.txt'), ['#lane_c=phiK[n,c]#', '#unsigned4#', '#IFIFO#'],
              [pack(phik[h, n].tolist(), 4) for n in range(N)])
        write(os.path.join(out_dir, f'h{h}_phiq.txt'), ['#lane_c=phiQ[n,c]#', '#unsigned4#', '#L0#'],
              [pack(phiq[h, n].tolist(), 4) for n in range(N)])
        write(os.path.join(out_dir, f'h{h}_S.txt'), ['#line_r=S[:,r]#', '#16b_lanes_c#', f'#sh={sh[h].item()}#'],
              [pack(S[h, :, r].tolist(), 16) for r in range(8)])
        write(os.path.join(out_dir, f'h{h}_s8hi.txt'), ['#line_t=S8[:,t]>>4#', '#lanes_c#', '#signed4#'],
              [pack(hi[h, :, t].tolist(), 4) for t in range(8)])
        write(os.path.join(out_dir, f'h{h}_s8lo.txt'), ['#line_t=S8[:,t]&15#', '#lanes_c#', '#unsigned4#'],
              [pack(lo[h, :, t].tolist(), 4) for t in range(8)])
        write(os.path.join(out_dir, f'h{h}_out.txt'), ['#lane_r<7=out#', '#lane7=den#', '#16b#'],
              [pack(out[h, n].tolist() + [P[h, n, D_V].item()], 16) for n in range(N)])
        stats.append('head {} sh {} max|S| {} max|P| {} zero phiQ {:.3f} phiK {:.3f} V {:.3f} den0 {}'.format(
            h, sh[h].item(), S[h].abs().max().item(), P[h].abs().max().item(), (phiq[h] == 0).float().mean().item(),
            (phik[h] == 0).float().mean().item(), (v[h] == 0).float().mean().item(),
            int((P[h, :, D_V] == 0).sum().item())))
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        f.write(f'{H} {N}\n')
    with open(os.path.join(out_dir, 'stats.txt'), 'w') as f:
        f.write(tag + '\n' + '\n'.join(stats) + '\n')
    return out


def from_model(layer, image, out_dir):
    import train_vit as T
    from vit import ViT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ViT().to(T.dev)
    T.load(model, 'linear_q.pth')
    model.set_mode('linear_q', hw_exact=True)
    model.eval()
    _, _, xt, yt = T.load_cifar()
    attn = model.blocks[layer].attn
    attn.capture = {}
    pre = {}
    hk = attn.proj.register_forward_pre_hook(lambda m, i: pre.__setitem__('x', i[0].detach()))
    with torch.no_grad():
        logits = model(T.norm(xt[image:image + 1]))
    hk.remove()
    cap = attn.capture
    attn.capture = None
    phiq, phik, v = cap['phiq'][0].cpu(), cap['phik'][0].cpu(), cap['v'][0].cpu()
    tag = f'layer {layer} image {image} label {yt[image].item()} pred {logits.argmax(1).item()}'
    out = export(out_dir, phiq, phik, v, tag)
    # check against the model: proj input [1, N, H*7] == out_int * s_v (per head)
    sv = attn.vq.scale().detach().cpu()
    H, N, _ = out.shape
    rec = (out.double() * sv.double().view(H, 1, 1)).transpose(0, 1).reshape(N, H * D_V)
    err = (rec - pre['x'][0].double().cpu()).abs().max().item()
    print(f'{tag}: max |out_hw * s_v - proj prehook| = {err:.2e}')
    assert err < 1e-5
    with open(os.path.join(out_dir, 'stats.txt'), 'a') as f:
        f.write(f'prehook_error {err:.3e}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--image', type=int, default=0)
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.random:
        g = torch.Generator().manual_seed(7)
        H, N = 4, 64
        phiq = torch.randint(0, 16, (H, N, 8), generator=g)
        phik = torch.randint(0, 16, (H, N, 8), generator=g)
        v = torch.randint(-8, 8, (H, N, 7), generator=g)
        phiq[0, :5] = 0          # tokens with den = 0
        phik[1, :, 3] = 0        # a dead key feature
        export(args.out, phiq, phik, v, 'random')
        print('random stimulus written to', args.out)
    else:
        from_model(args.layer, args.image, args.out)


if __name__ == '__main__':
    main()

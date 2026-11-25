"""Part 1 Step 6 (+alpha): stimulus for any 3x3 QuantConv2d of the squeezed VGG16 on the 8x8 array, with tiling.

    features[24]: 256 -> 8   : 32 input-channel tiles x 1 output tile   (partial sums accumulated in the SFU)
    features[27]:   8 -> 8   :  1 x 1                                   (the squeezed layer, + ReLU)
    features[29]:   8 -> 512 :  1 x 64 output-channel tiles

Files in --out:
  config.txt                              n_itile n_otile relu len_nij len_onij
  activation_tile{it}.txt                 36 x 32b (padded 6x6, ic 8*it+7 ... 8*it)
  weight_itile{it}_otile{ot}_kij{k}.txt    8 x 32b (line t = oc 8*ot+t)
  acc_address.txt                        9 lines per o_nij (same layout as gen_stimulus.py)
  out_otile{ot}.txt                      16 x 128b: final integer output of the layer for oc 8*ot .. 8*ot+7
                                         (sum over all ic tiles and kij; ReLU applied if relu = 1)
The integer outputs are checked against the float layer: conv output (forward hook) * 1/(x_delta*w_delta),
and for features[27] also ReLU(out) against the pre-hooked input of features[29].

Usage (from project/software/):
    python scripts/gen_tiled_stimulus.py --layer 24 --image 0 --out ../hardware/sim/stim_f24
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SW)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net.vgg_quant_squeeze import VGG16_quant_squeeze, SQUEEZED_IDX, NEXT_IDX  # noqa: E402
from models.quant_layer import act_quantization  # noqa: E402
from gen_stimulus import pack, write, ARRAY, BW, PSUM_BW  # noqa: E402


def load_model_and_data(ckpt, device):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    test = torchvision.datasets.CIFAR10(root=os.path.join(SW, 'data'), train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    model = VGG16_quant_squeeze()
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    return model.to(device).eval(), test


def layer_case(model, image, layer_idx, device):
    layer = model.features[layer_idx]
    saved = {}
    hooks = [layer.register_forward_pre_hook(lambda m, i: saved.__setitem__('x', i[0].detach().cpu())),
             layer.register_forward_hook(lambda m, i, o: saved.__setitem__('y', o.detach().cpu()))]
    if layer_idx == SQUEEZED_IDX:
        hooks.append(model.features[NEXT_IDX].register_forward_pre_hook(
            lambda m, i: saved.__setitem__('next', i[0].detach().cpu())))
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
        weight_q = layer.weight_quant(layer.weight).detach().cpu()
    for h in hooks:
        h.remove()
    w_delta = layer.weight_quant.wgt_alpha.detach().cpu() / 7
    a_alpha = layer.act_alpha.detach().cpu()
    x_delta = a_alpha / 15
    w_int = (weight_q / w_delta).round()
    x_int = (act_quantization(4)(saved['x'][0], a_alpha) / x_delta).round()
    assert ((weight_q / w_delta) - w_int).abs().max() < 1e-3
    assert w_int.abs().max() <= 7 and 0 <= x_int.min() and x_int.max() <= 15
    return x_int, w_int, x_delta, w_delta, saved


def generate(model, test, image_idx, layer_idx, out_dir, device, quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    image, label = test[image_idx]
    x_int, w_int, x_delta, w_delta, saved = layer_case(model, image, layer_idx, device)
    relu = int(layer_idx == SQUEEZED_IDX)     # only the squeezed conv is followed directly by ReLU
    C_out, C_in, k, _ = w_int.shape
    n_it, n_ot = C_in // ARRAY, C_out // ARRAY
    assert n_it * ARRAY == C_in and n_ot * ARRAY == C_out

    H = x_int.shape[1]
    P = H + 2
    a_pad = torch.zeros(C_in, P, P)
    a_pad[:, 1:1 + H, 1:1 + H] = x_int
    a_pad = a_pad.reshape(C_in, -1).long()
    n_nij = a_pad.shape[1]
    O = P - k + 1
    w_int = w_int.long()

    for it in range(n_it):
        write(os.path.join(out_dir, 'activation_tile{}.txt'.format(it)),
              ['#row7[msb]...row0[lsb]#', '#line=nij#', '#tile{}#'.format(it)],
              [pack(a_pad[ARRAY * it:ARRAY * (it + 1), nij].tolist(), BW, False) for nij in range(n_nij)])

    max_raw = max_partial = 0
    out = torch.zeros(n_ot, O * O, ARRAY, dtype=torch.long)
    for ot in range(n_ot):
        partial = torch.zeros(O * O, ARRAY, dtype=torch.long)
        for it in range(n_it):
            a_t = a_pad[ARRAY * it:ARRAY * (it + 1)]
            for kij in range(k * k):
                ky, kx = divmod(kij, k)
                w_t = w_int[ARRAY * ot:ARRAY * (ot + 1), ARRAY * it:ARRAY * (it + 1), ky, kx]   # [oc, ic]
                write(os.path.join(out_dir, 'weight_itile{}_otile{}_kij{}.txt'.format(it, ot, kij)),
                      ['#col_t=oc#', '#ic7[msb]...ic0[lsb]#', '#4b_signed#'],
                      [pack(w_t[t].tolist(), BW, True) for t in range(ARRAY)])
                raw = (w_t @ a_t).t()                                   # [nij, oc] what the array produces
                max_raw = max(max_raw, raw.abs().max().item())
                for o in range(O * O):
                    oy, ox = divmod(o, O)
                    partial[o] += raw[(oy + ky) * P + (ox + kx)]
            max_partial = max(max_partial, partial.abs().max().item())   # what the SFU/pmem holds after tile it
        out[ot] = partial
    assert max_raw < 2 ** (PSUM_BW - 1) and max_partial < 2 ** (PSUM_BW - 1), 'psum overflows 16 bits'

    final = out.clamp(min=0) if relu else out
    for ot in range(n_ot):
        write(os.path.join(out_dir, 'out_otile{}.txt'.format(ot)),
              ['#relu={}#'.format(relu), '#col7[msb]...col0[lsb]#', '#line=o_nij#'],
              [pack(final[ot, o].tolist(), PSUM_BW, True) for o in range(O * O)])

    addr = []
    for o in range(O * O):
        oy, ox = divmod(o, O)
        for kij in range(k * k):
            ky, kx = divmod(kij, k)
            addr.append(format(kij * n_nij + (oy + ky) * P + (ox + kx), '011b'))
    write(os.path.join(out_dir, 'acc_address.txt'), ['#9_lines_per_o_nij#', '#addr=kij*36+nij#', '#11b#'], addr)
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        f.write('{} {} {} {} {}\n'.format(n_it, n_ot, relu, n_nij, O * O))

    # ---- checks against the float model ----
    out_cf = out.permute(0, 2, 1).reshape(C_out, O * O)                            # [C_out, o_nij]
    ref = F.conv2d(x_int.unsqueeze(0), w_int.float(), padding=1)[0].reshape(C_out, -1).long()
    assert torch.equal(ref, out_cf)
    conv_err = (out_cf.float() * x_delta * w_delta - saved['y'][0].reshape(C_out, -1)).abs().max().item()
    msg = 'features[{}] image {} (label {}): {} itile x {} otile, relu={} | max|raw| {} max|partial| {} | ' \
          'conv-output error {:.2e}'.format(layer_idx, image_idx, label, n_it, n_ot, relu, max_raw, max_partial,
                                            conv_err)
    assert conv_err < 1e-3
    if relu:
        nxt_err = (final.permute(0, 2, 1).reshape(C_out, -1).float() * x_delta * w_delta
                   - saved['next'][0].reshape(C_out, -1)).abs().max().item()
        msg += ' | next-layer prehook error {:.2e}'.format(nxt_err)
        assert nxt_err < 1e-3
    if not quiet:
        print(msg)
    return msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', type=int, default=SQUEEZED_IDX)
    p.add_argument('--image', type=int, default=0)
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, test = load_model_and_data(args.ckpt, device)
    generate(model, test, args.image, args.layer, args.out, device)


if __name__ == '__main__':
    main()

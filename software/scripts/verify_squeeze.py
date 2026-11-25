"""Part 1 Step 1 verification: 8x8 systolic-array emulation of the squeezed features[27] and psum_recovered check.

  1. test accuracy of the trained squeezed model (> 90% required)
  2. pre-hook features[27] (input of the squeezed conv) and features[29] (input of the next layer)
  3. integer weights w_int in [-7, 7] and activations x_int in [0, 15]
  4. 8x8 array emulation (no tiling: 8 ic rows x 8 oc columns), psum[oc, nij, kij], then accumulation over kij
  5. psum_recovered = ReLU(out_int * x_delta * w_delta), compared against the pre-hooked input of features[29]

Usage (from project/software/):
    python scripts/verify_squeeze.py [--ckpt result/vgg16_quant_4bit_squeeze8/model_best.pth.tar]
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
from net.vgg_quant_squeeze import VGG16_quant_squeeze, SQUEEZED_IDX, NEXT_IDX  # noqa: E402
from models.quant_layer import act_quantization  # noqa: E402

ARRAY = 8
W_BIT, A_BIT = 4, 4


class SaveInput:
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in):
        self.inputs.append(module_in[0].detach().clone())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=os.path.join(SW, 'result', 'vgg16_quant_4bit_squeeze8', 'model_best.pth.tar'))
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ampere GPUs run cuDNN convs in TF32 (10-bit mantissa) by default; use true FP32 for the reference
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    test = torchvision.datasets.CIFAR10(root=os.path.join(SW, 'data'), train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    testloader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=2)

    model = VGG16_quant_squeeze()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    # ---- 1. accuracy ----
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print('checkpoint epoch {} | test accuracy = {:.2f}%'.format(ckpt['epoch'], acc))
    print('features[{}] = {}  ->  features[{}] = {}'.format(
        SQUEEZED_IDX, model.features[SQUEEZED_IDX], SQUEEZED_IDX + 1, model.features[SQUEEZED_IDX + 1]))

    # ---- 2. pre-hooks ----
    layer, nxt = model.features[SQUEEZED_IDX], model.features[NEXT_IDX]
    h_in, h_next = SaveInput(), SaveInput()
    hooks = [layer.register_forward_pre_hook(h_in), nxt.register_forward_pre_hook(h_next)]
    images, _ = next(iter(testloader))
    with torch.no_grad():
        model(images.to(device))
    for h in hooks:
        h.remove()

    # ---- 3. integer weights / activations ----
    w_alpha = layer.weight_quant.wgt_alpha.detach().cpu()
    a_alpha = layer.act_alpha.detach().cpu()
    w_delta = w_alpha / (2 ** (W_BIT - 1) - 1)
    x_delta = a_alpha / (2 ** A_BIT - 1)

    with torch.no_grad():
        weight_q = layer.weight_quant(layer.weight).detach().cpu()   # same op as in QuantConv2d.forward
    w_float = weight_q / w_delta
    w_int = w_float.round()

    x = h_in.inputs[0][0].cpu()                                      # image 0, [8, 4, 4]
    x_q = act_quantization(A_BIT)(x, a_alpha)
    x_float = x_q / x_delta
    x_int = x_float.round()

    print('wgt_alpha = {:.4f} (w_delta {:.4f}) | act_alpha = {:.4f} (x_delta {:.4f})'.format(
        w_alpha.item(), w_delta.item(), a_alpha.item(), x_delta.item()))
    print('w_int {} range [{:.0f}, {:.0f}] | x_int {} range [{:.0f}, {:.0f}]'.format(
        tuple(w_int.shape), w_int.min(), w_int.max(), tuple(x_int.shape), x_int.min(), x_int.max()))
    assert (w_float - w_int).abs().max() < 1e-4 and (x_float - x_int).abs().max() < 1e-4
    assert w_int.abs().max() <= 7 and x_int.min() >= 0 and x_int.max() <= 15

    # ---- 4. 8x8 systolic array emulation (weight stationary, no tiling) ----
    C_out, C_in, k, _ = w_int.shape
    assert C_out == C_in == ARRAY
    n_kij = k * k
    pad = 1
    H, W = x_int.shape[1:]
    P = H + 2 * pad
    a_pad = torch.zeros(C_in, P, P)
    a_pad[:, pad:pad + H, pad:pad + W] = x_int
    a_pad = a_pad.reshape(C_in, -1)                                  # [8 rows, nij = 36]
    n_nij = a_pad.shape[1]

    w_kij = w_int.permute(2, 3, 1, 0).reshape(n_kij, C_in, C_out)   # [kij, row = ic, col = oc]
    psum = torch.zeros(C_out, n_nij, n_kij)
    for kij in range(n_kij):
        for nij in range(n_nij):                                     # one padded input pixel per time step
            psum[:, nij, kij] = w_kij[kij].t() @ a_pad[:, nij]

    O = P - k + 1
    out_int = torch.zeros(C_out, O * O)
    for o_nij in range(O * O):
        oy, ox = divmod(o_nij, O)
        for kij in range(n_kij):
            ky, kx = divmod(kij, k)
            out_int[:, o_nij] += psum[:, (oy + ky) * P + (ox + kx), kij]

    ref_int = F.conv2d(x_int.unsqueeze(0), w_int, padding=pad).reshape(C_out, -1)
    print('array emulation: nij = {}, kij = {}, o_nij = {} | max |out_int - conv(x_int, w_int)| = {:.0f}'.format(
        n_nij, n_kij, O * O, (out_int - ref_int).abs().max()))
    print('out_int range [{:.0f}, {:.0f}]'.format(out_int.min(), out_int.max()))

    # ---- 5. psum_recovered vs. pre-hooked input of the next layer ----
    psum_recovered = F.relu(out_int * x_delta * w_delta)            # [8, 16]
    next_in = h_next.inputs[0][0].cpu().reshape(C_out, -1)
    err = (psum_recovered - next_in).abs()
    print('max |psum_recovered - prehook(features[{}])| = {:.3e}  (mean {:.3e})'.format(
        NEXT_IDX, err.max().item(), err.mean().item()))

    # whole batch, to make sure image 0 is not a lucky case
    xb = h_in.inputs[0].cpu()
    xb_int = (act_quantization(A_BIT)(xb, a_alpha) / x_delta).round()
    ob = F.relu(F.conv2d(xb_int, w_int, padding=pad) * x_delta * w_delta)
    errb = (ob - h_next.inputs[0].cpu()).abs().max().item()
    print('whole batch ({} images): max error = {:.3e}'.format(xb.shape[0], errb))

    ok = acc > 90.0 and err.max().item() < 1e-3
    print('PASS' if ok else 'FAIL', '(accuracy > 90% and error < 1e-3)')


if __name__ == '__main__':
    main()

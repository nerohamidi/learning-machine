"""Train the CIFAR-10 tiny ViT and turn its softmax attention into hardware linear attention.

Stages (each loads the previous checkpoint from result/vit/):
  teacher   softmax attention, float                                    -> teacher.pth
  transfer  linear attention (phi = ReLU), attention transfer: only W_q / W_k (+bias) are trained so that
            every layer's linear-attention output matches the teacher's softmax output on the teacher's own
            layer input (MSE, LoLCATs-style)                             -> transfer.pth
  finetune  linear attention, float, all weights, CE + KD from the teacher -> linear.pth
  qat       4-bit QuantLinear + quantised attention core (fake quant), CE + KD -> linear_q.pth
  eval      accuracy table for every stage, incl. the bit-exact integer attention core (hw_exact)

Usage (from project/software/transformer, conda env ml):
    python train_vit.py --stage teacher --epochs 150
    python train_vit.py --stage transfer --epochs 5
    python train_vit.py --stage finetune --epochs 40
    python train_vit.py --stage qat --epochs 40
    python train_vit.py --stage eval
"""

import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import ViT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'cifar-10-batches-py')
OUT = os.path.join(HERE, '..', 'result', 'vit')
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
dev = 'cuda'


def load_cifar():
    def rd(f):
        with open(os.path.join(DATA, f), 'rb') as fh:
            d = pickle.load(fh, encoding='bytes')
        return np.asarray(d[b'data'], np.uint8).reshape(-1, 3, 32, 32), np.asarray(d[b'labels'])
    xs, ys = zip(*[rd(f'data_batch_{i}') for i in range(1, 6)])
    xt, yt = rd('test_batch')
    t = lambda a: torch.from_numpy(a).to(dev)
    return t(np.concatenate(xs)), t(np.concatenate(ys)), t(xt), t(yt)


def norm(xb):
    return (xb.float() / 255 - MEAN.to(dev)) / STD.to(dev)


def augment(xb):
    """random crop (pad 4) + horizontal flip, on the GPU."""
    B = xb.shape[0]
    xp = F.pad(xb, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (B,), device=dev)
    dy = torch.randint(0, 9, (B,), device=dev)
    ar = torch.arange(32, device=dev)
    iy = (dy[:, None] + ar[None]).view(B, 1, 32, 1).expand(B, 3, 32, 40)
    ix = (dx[:, None] + ar[None]).view(B, 1, 1, 32).expand(B, 3, 32, 32)
    out = xp.gather(2, iy).gather(3, ix)
    flip = torch.rand(B, device=dev) < 0.5
    out[flip] = out[flip].flip(3)
    return out


@torch.no_grad()
def evaluate(model, xt, yt, bs=500, amp=True):
    model.eval()
    correct = 0
    for i in range(0, len(xt), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            correct += (model(norm(xt[i:i + bs])).argmax(1) == yt[i:i + bs]).sum().item()
    return 100.0 * correct / len(xt)


def ckpt(name):
    return os.path.join(OUT, name)


def save(model, name, info):
    torch.save({'state_dict': model.state_dict(), 'info': info}, ckpt(name))


def load(model, name):
    sd = torch.load(ckpt(name), map_location=dev)
    model.load_state_dict(sd['state_dict'])
    return sd['info']


def train(model, data, epochs, lr, teacher=None, amp=True, params=None, wd=0.05, bs=256, log=None):
    x, y, xt, yt = data
    params = list(params if params is not None else model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps = epochs * (len(x) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1 if epochs > 10 else 0.2)
    best, best_sd = -1, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(x), device=dev)
        tot = 0.0
        for i in range(len(x) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            xb = norm(augment(x[idx]))
            yb = y[idx]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                out = model(xb)
                loss = F.cross_entropy(out.float(), yb, label_smoothing=0.1)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(xb).float()
                    T = 2.0
                    loss = 0.5 * loss + 0.5 * T * T * F.kl_div(F.log_softmax(out.float() / T, 1),
                                                               F.softmax(tl / T, 1), reduction='batchmean')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        acc = evaluate(model, xt, yt, amp=amp)
        if acc > best:
            best, best_sd = acc, copy.deepcopy(model.state_dict())
        msg = f'ep {ep:3d} loss {tot / (len(x) // bs):.4f} test {acc:.2f} best {best:.2f} ({time.time() - t0:.1f}s)'
        print(msg, flush=True)
        if log:
            log.write(msg + '\n')
            log.flush()
    model.load_state_dict(best_sd)
    return best


def attention_transfer(student, teacher, data, epochs, lr=2e-3, bs=256, log=None):
    """Only q/k projections of the student are trained; loss = sum over layers of
    MSE(linear_attn(x_l), softmax_attn(x_l)) with x_l the teacher's normalised input of layer l."""
    x = data[0]
    params = [p for a in student.attns() for m in (a.q, a.k) for p in m.parameters() if p.requires_grad]
    params = [p for p in params if p.dim() >= 1]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * (len(x) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    feats = {}
    hooks = [b.attn.register_forward_pre_hook(lambda m, inp, i=i: feats.__setitem__(i, inp[0].detach()))
             for i, b in enumerate(teacher.blocks)]
    teacher.eval()
    student.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=dev)
        tot = 0.0
        for i in range(len(x) // bs):
            xb = norm(augment(x[perm[i * bs:(i + 1) * bs]]))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                teacher(xb)
                tgt = {l: teacher.blocks[l].attn.core(feats[l]).float() for l in feats}
            loss = 0.0
            for l, a in enumerate(student.attns()):
                loss = loss + F.mse_loss(a.core(feats[l].float()), tgt[l])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f'transfer ep {ep} mse(sum over layers) {tot / (len(x) // bs):.5f}'
        print(msg, flush=True)
        if log:
            log.write(msg + '\n')
    for h in hooks:
        h.remove()


@torch.no_grad()
def attn_fidelity(student, teacher, xt, n=1000):
    """cosine similarity between student (linear) and teacher (softmax) attention outputs, same layer input."""
    feats = {}
    hooks = [b.attn.register_forward_pre_hook(lambda m, inp, i=i: feats.__setitem__(i, inp[0].detach()))
             for i, b in enumerate(teacher.blocks)]
    teacher.eval(); student.eval()
    teacher(norm(xt[:n]))
    for h in hooks:
        h.remove()
    cs = []
    for l, a in enumerate(student.attns()):
        t = teacher.blocks[l].attn.core(feats[l]).flatten(1)
        s = a.core(feats[l]).flatten(1)
        cs.append(F.cosine_similarity(s, t, dim=1).mean().item())
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['teacher', 'transfer', 'finetune', 'qat', 'eval'])
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    data = load_cifar()
    log = open(os.path.join(OUT, f'log_{args.stage}.txt'), 'a')
    model = ViT().to(dev)

    if args.stage == 'teacher':
        model.set_mode('softmax')
        best = train(model, data, args.epochs or 150, args.lr or 1e-3, log=log)
        save(model, 'teacher.pth', {'acc': best})

    elif args.stage == 'transfer':
        teacher = ViT().to(dev); load(teacher, 'teacher.pth'); teacher.set_mode('softmax')
        load(model, 'teacher.pth')
        model.set_mode('linear')
        acc0 = evaluate(model, data[2], data[3])
        fid0 = attn_fidelity(model, teacher, data[2])
        attention_transfer(model, teacher, data, args.epochs or 5, lr=args.lr or 2e-3, log=log)
        acc1 = evaluate(model, data[2], data[3])
        fid1 = attn_fidelity(model, teacher, data[2])
        info = {'swap_acc': acc0, 'swap_cos': fid0, 'transfer_acc': acc1, 'transfer_cos': fid1}
        print(json.dumps(info, indent=1)); log.write(json.dumps(info) + '\n')
        save(model, 'transfer.pth', info)

    elif args.stage == 'finetune':
        teacher = ViT().to(dev); load(teacher, 'teacher.pth'); teacher.set_mode('softmax'); teacher.eval()
        load(model, 'transfer.pth'); model.set_mode('linear')
        best = train(model, data, args.epochs or 40, args.lr or 5e-4, teacher=teacher, log=log)
        save(model, 'linear.pth', {'acc': best, 'cos': attn_fidelity(model, teacher, data[2])})

    elif args.stage == 'qat':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        teacher = ViT().to(dev); load(teacher, 'teacher.pth'); teacher.set_mode('softmax'); teacher.eval()
        load(model, 'linear.pth'); model.set_mode('linear_q')
        model.train()
        with torch.no_grad():   # LSQ scale initialisation on one batch
            model(norm(data[0][:500]))
        print('fake-quant acc before QAT', evaluate(model, data[2], data[3], amp=False))
        best = train(model, data, args.epochs or 40, args.lr or 2e-4, teacher=teacher, amp=False, wd=0.0, log=log)
        save(model, 'linear_q.pth', {'acc': best})

    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        teacher = ViT().to(dev); ti = load(teacher, 'teacher.pth'); teacher.set_mode('softmax')
        res = {'teacher_softmax_float': evaluate(teacher, data[2], data[3], amp=False)}
        m = ViT().to(dev); load(m, 'teacher.pth'); m.set_mode('linear')
        res['swap_to_linear_no_training'] = evaluate(m, data[2], data[3], amp=False)
        ti = load(m, 'transfer.pth'); m.set_mode('linear')
        res['after_attention_transfer'] = evaluate(m, data[2], data[3], amp=False)
        res['transfer_cos_before_after'] = [ti['swap_cos'], ti['transfer_cos']]
        load(m, 'linear.pth'); m.set_mode('linear')
        res['linear_float_finetuned'] = evaluate(m, data[2], data[3], amp=False)
        res['linear_float_cos_to_teacher'] = attn_fidelity(m, teacher, data[2])
        load(m, 'linear_q.pth'); m.set_mode('linear_q')
        res['linear_4bit_fakequant'] = evaluate(m, data[2], data[3], amp=False)
        m.set_mode('linear_q', hw_exact=True)
        res['linear_4bit_hw_exact'] = evaluate(m, data[2], data[3], amp=False)
        print(json.dumps(res, indent=1))
        with open(os.path.join(OUT, 'eval.json'), 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()

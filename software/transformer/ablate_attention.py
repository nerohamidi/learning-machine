"""Ablation: does the tiny CIFAR-10 ViT actually use attention?

Trains the same ViT with the same recipe as the softmax teacher (train_vit.py --stage teacher: float, AdamW,
OneCycle lr 1e-3, 120 epochs, crop+flip) from scratch, with the token mixer replaced:
  softmax   the normal teacher (seed-matched rerun, the reference)
  none      attention branch removed: x + 0. Tokens never interact until the final mean-pool before the head.
  uniform   every token receives the mean of all value vectors (attention matrix fixed to 1/N): token mixing
            exists but is not content-dependent. Same parameter count as softmax (W_q, W_k unused).

Usage (from project/software/transformer, conda env ml):
    python ablate_attention.py --mixer softmax|none|uniform [--epochs 120] [--seed 0]
Results: result/vit/ablation/log_<mixer>.txt and result/vit/ablation/<mixer>.json
"""

import argparse
import json
import os

import torch

import vit
from vit import ViT
from train_vit import OUT, dev, load_cifar, train


class NoMixer(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def uniform_core(self, x):
    B, N, _ = x.shape
    v = self.v(x).view(B, N, self.h, vit.D_V).transpose(1, 2)
    return v.mean(2, keepdim=True).expand(-1, -1, N, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mixer', required=True, choices=['softmax', 'none', 'uniform'])
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out = os.path.join(OUT, 'ablation')
    os.makedirs(out, exist_ok=True)
    model = ViT().to(dev)
    model.set_mode('softmax')
    if args.mixer == 'none':
        for b in model.blocks:
            b.attn = NoMixer()
    elif args.mixer == 'uniform':
        vit.Attention.core = uniform_core
    n_param = sum(p.numel() for p in model.parameters())
    with open(os.path.join(out, f'log_{args.mixer}.txt'), 'w') as log:
        log.write(f'mixer {args.mixer} epochs {args.epochs} seed {args.seed} params {n_param}\n')
        best = train(model, load_cifar(), args.epochs, 1e-3, log=log)
    res = dict(mixer=args.mixer, epochs=args.epochs, seed=args.seed, params=n_param, best_test=best)
    print(res)
    with open(os.path.join(out, f'{args.mixer}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()

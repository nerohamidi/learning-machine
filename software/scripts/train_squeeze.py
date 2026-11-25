"""Part 1 Step 1: 4-bit/4-bit QAT of VGG16 with features[27] squeezed to 8->8 channels (BatchNorm removed).

Initialised from the HW6 4-bit VGG16_quant checkpoint (91.61%). Layers whose shapes are unchanged are copied
directly. The three layers around the squeeze keep the 8 channels of the old features[24] with the largest
|BN gamma|, so they start from a slice of the trained weights instead of from random values:
    new features[24] (256->8)  <- old features[24][sel, :]      + BN[25][sel]
    new features[27] (8->8)    <- old features[27][sel][:, sel]  (old BN[28] is dropped)
    new features[29] (8->512)  <- old features[30][:, sel]      + BN[31] -> BN[30]
All later layers shift down by one index because the BatchNorm after the squeezed conv is removed.

Usage (from project/software/):
    python scripts/train_squeeze.py --epochs 60
"""

import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = os.path.join(SW, '..', '..')
sys.path.insert(0, SW)
from net.vgg_quant_squeeze import VGG16_quant_squeeze, SQUEEZE_CH  # noqa: E402
from models.quant_layer import QuantConv2d  # noqa: E402  (course package, path added by net/)

INIT = os.path.join(PREP, 'hw6', 'result', 'vgg16_quant_4bit', 'model_best.pth.tar')
DATA = os.path.join(SW, 'data')


def conv_bn_pairs(model):
    """[(conv, bn or None)] in network order."""
    mods, pairs = list(model.features), []
    for i, m in enumerate(mods):
        if isinstance(m, QuantConv2d):
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            pairs.append((m, nxt if isinstance(nxt, nn.BatchNorm2d) else None))
    return pairs


def init_from_hw6(new, path):
    from models import VGG16_quant
    old = VGG16_quant()
    sd = torch.load(path, map_location='cpu')['state_dict']
    old.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})

    op, np_ = conv_bn_pairs(old), conv_bn_pairs(new)
    assert len(op) == len(np_) == 13
    sq = 8  # conv #8 (0-based) is features[27] (convs at 0,3,7,10,14,17,20,24,27,...)
    sel = op[sq - 1][1].weight.detach().abs().argsort(descending=True)[:SQUEEZE_CH].sort().values

    def copy_conv(dst, src, w):
        dst.weight.data.copy_(w)
        dst.act_alpha.data.copy_(src.act_alpha.data)
        dst.weight_quant.wgt_alpha.data.copy_(src.weight_quant.wgt_alpha.data)

    def copy_bn(dst, src, idx=None):
        for name in ['weight', 'bias', 'running_mean', 'running_var']:
            t = getattr(src, name).data
            getattr(dst, name).data.copy_(t if idx is None else t[idx])

    with torch.no_grad():
        for i, ((oc, ob), (nc, nb)) in enumerate(zip(op, np_)):
            if i == sq - 1:
                copy_conv(nc, oc, oc.weight[sel])
                copy_bn(nb, ob, sel)
            elif i == sq:
                copy_conv(nc, oc, oc.weight[sel][:, sel])
                assert nb is None
            elif i == sq + 1:
                copy_conv(nc, oc, oc.weight[:, sel])
                copy_bn(nb, ob)
            else:
                copy_conv(nc, oc, oc.weight)
                copy_bn(nb, ob)
        new.classifier.load_state_dict(old.classifier.state_dict())
    return sel.tolist()


def get_loaders(batch_size):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train = torchvision.datasets.CIFAR10(root=DATA, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    test = torchvision.datasets.CIFAR10(root=DATA, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), normalize]))
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4,
                                        persistent_workers=True),
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2,
                                        persistent_workers=True))


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def validate(loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--name', default='vgg16_quant_4bit_squeeze8')
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fdir = os.path.join(SW, 'result', args.name)
    os.makedirs(fdir, exist_ok=True)
    log = open(os.path.join(fdir, 'train_log.txt'), 'a')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    model = VGG16_quant_squeeze()
    sel = init_from_hw6(model, INIT)
    say('init from {} | kept channels {}'.format(os.path.relpath(INIT, SW), sel))
    model.to(device)

    trainloader, testloader = get_loaders(args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    say(str(vars(args)))
    say('start acc (after squeeze, before QAT): {:.2f}%'.format(validate(testloader, model, device)))

    best_prec = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(trainloader, model, criterion, optimizer, device)
        scheduler.step()
        prec = validate(testloader, model, device)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        say('epoch {:3d} | lr {:.5f} | train loss {:.4f} acc {:.2f}% | test {:.2f}% | best {:.2f}% | {:.0f}s'.format(
            epoch, optimizer.param_groups[0]['lr'], tr_loss, tr_acc, prec, best_prec, time.time() - t0))
        ckpt = os.path.join(fdir, 'checkpoint.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec': best_prec,
                    'args': vars(args)}, ckpt)
        if is_best:
            shutil.copyfile(ckpt, os.path.join(fdir, 'model_best.pth.tar'))
    say('done, best {:.2f}%'.format(best_prec))


if __name__ == '__main__':
    main()

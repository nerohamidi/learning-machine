"""VGG16_quant with one convolution squeezed to 8 input / 8 output channels for the 8x8 systolic array.

Layout of `features` (indices matter for the hooks in the verification notebook):

    [24] QuantConv2d(256 -> 8)   [25] BatchNorm2d(8)   [26] ReLU
    [27] QuantConv2d(8 -> 8)     [28] ReLU                         <- squeezed layer, no BatchNorm
    [29] QuantConv2d(8 -> 512)   [30] BatchNorm2d(512) [31] ReLU   <- its pre-hooked input = ReLU(features[27](x))

Every other layer is identical to the course VGG16_quant. The squeezed conv runs on the 4x4 feature map
(after three max-pools), so one image is only 16 output pixels (36 padded input pixels) on the array.
"""

import os
import sys

import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ece284fa26', 'software'))
from models.quant_layer import QuantConv2d  # course quantizer, unchanged

SQUEEZE_CH = 8

# (out_channels, keep_batchnorm) per conv, 'M' = max-pool.
CFG = [(64, True), (64, True), 'M',
       (128, True), (128, True), 'M',
       (256, True), (256, True), (256, True), 'M',
       (SQUEEZE_CH, True), (SQUEEZE_CH, False), (512, True), 'M',
       (512, True), (512, True), (512, True), 'M']

SQUEEZED_IDX = 27   # features[27]: QuantConv2d(8, 8), followed directly by ReLU
NEXT_IDX = 29       # features[29]: first layer after the squeezed conv + ReLU


class VGG16_quant_squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        layers, in_ch = [], 3
        for item in CFG:
            if item == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            out_ch, keep_bn = item
            layers += [QuantConv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if keep_bn:
                layers += [nn.BatchNorm2d(out_ch)]
            layers += [nn.ReLU(inplace=True)]
            in_ch = out_ch
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 10)

        assert isinstance(self.features[SQUEEZED_IDX], QuantConv2d)
        assert self.features[SQUEEZED_IDX].in_channels == SQUEEZE_CH == self.features[SQUEEZED_IDX].out_channels
        assert isinstance(self.features[SQUEEZED_IDX + 1], nn.ReLU)
        assert isinstance(self.features[NEXT_IDX], QuantConv2d)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()

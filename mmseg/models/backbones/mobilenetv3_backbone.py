import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule


class Hswish(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3, inplace=True) / 6


class Hsigmoid(nn.Module):
    def forward(self, x):
        return nn.functional.relu6(x + 3, inplace=True) / 6


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_channels, kernel_size, stride, se, nl, dilation=1):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        act = Hswish() if nl == 'HS' else nn.ReLU(inplace=True)

        layers = [
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            act,
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride,
                      padding=(kernel_size - 1) // 2 * dilation,
                      dilation=dilation,
                      groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            SEModule(exp_channels) if se else nn.Identity(),
            act,
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


@MODELS.register_module()
class MobileNetV3Backbone(BaseModule):
    def __init__(self, mode='small', width_mult=1.0, out_indices=(1, 4), init_cfg=None):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.mode = mode

        input_channel = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            Hswish()
        )

        layers = []
        if mode == 'small':
            cfg = [
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        else:
            raise ValueError("Only 'small' mode is currently supported")

        self.in_channels_list = []
        layers_list = []
        for idx, (k, exp, c, se, nl, s) in enumerate(cfg):
            out_c = int(c * width_mult)
            exp_c = int(exp * width_mult)
            layers_list.append(Bottleneck(input_channel, out_c, exp_c, k, s, se, nl))
            input_channel = out_c
            self.in_channels_list.append(out_c)

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

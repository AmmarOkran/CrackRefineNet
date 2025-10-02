import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class DeepCrackBackbone(BaseModule):
    def __init__(self, in_channels=3, base_channels=32, norm_type='BN', init_cfg=None):
        super().__init__(init_cfg)
        self.norm_layer = self._get_norm_layer(norm_type)

        self.conv1 = nn.Sequential(*self._conv_block(in_channels, base_channels, num_block=2))
        self.conv2 = nn.Sequential(*self._conv_block(base_channels, base_channels * 2, num_block=2))
        self.conv3 = nn.Sequential(*self._conv_block(base_channels * 2, base_channels * 4, num_block=3))
        self.conv4 = nn.Sequential(*self._conv_block(base_channels * 4, base_channels * 8, num_block=3))
        self.conv5 = nn.Sequential(*self._conv_block(base_channels * 8, base_channels * 8, num_block=3))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _get_norm_layer(self, norm_type):
        if norm_type.lower() == 'bn':
            return nn.BatchNorm2d
        elif norm_type.lower() == 'in':
            return nn.InstanceNorm2d
        elif norm_type.lower() == 'none':
            return lambda num_features: nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def _conv_block(self, in_nc, out_nc, num_block=2, kernel_size=3, stride=1, padding=1, bias=False):
        layers = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            layers += [
                nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                self.norm_layer(out_nc),
                nn.ReLU(inplace=True)
            ]
        return layers

    def forward(self, x):
        conv1 = self.conv1(x)                     # shape: (B, C, H, W)
        conv2 = self.conv2(self.maxpool(conv1))   # shape: (B, 2C, H/2, W/2)
        conv3 = self.conv3(self.maxpool(conv2))   # shape: (B, 4C, H/4, W/4)
        conv4 = self.conv4(self.maxpool(conv3))   # shape: (B, 8C, H/8, W/8)
        conv5 = self.conv5(self.maxpool(conv4))   # shape: (B, 8C, H/16, W/16)
        return [conv1, conv2, conv3, conv4, conv5]

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..utils.mobilevit import MobileViTBlock
from ..utils.dcn import DeformConv2d

from ..builder import HEADS
from .decode_head import BaseDecodeHead



def shortcut(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class DUC(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hdc = nn.Sequential(
                ConvBlock(in_channels, out_channels, padding=1, dilation=1),
                ConvBlock(out_channels, out_channels, padding=2, dilation=2),
                ConvBlock(out_channels, out_channels, padding=5, dilation=5, with_nonlinearity=False)
            )
        self.shortcut = shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.se = SE_Block(c=out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.se(self.hdc(x))
        x   = self.relu(res + x)
        return x


class DownBlockwithVit(nn.Module):
    def __init__(self, in_channels, out_channels, dim, L, kernel_size=3, patch_size=(4,4)):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.convblock  = ResidualBlock(in_channels, out_channels)
        self.vitblock   = MobileViTBlock(dim, L, out_channels, kernel_size, patch_size, int(dim*2))

    def forward(self, x):
        x = self.downsample(x)
        x = self.convblock(x)
        x = self.vitblock(x) + x
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.bridge = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        return self.bridge(self.downsample(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = DUC(in_channels, in_channels*2)
        self.residualblock = ResidualBlock(in_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.residualblock(x)
        return x    


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x
    
@HEADS.register_module()
class TransHead(BaseDecodeHead):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.out_channels = kwargs['out_channels']

        up_blocks = []

        self.bridge = Bridge(256, 512)

        up_blocks.append(UpBlock(in_channels=256*2, out_channels=256))
        up_blocks.append(UpBlock(in_channels=128*2, out_channels=128))
        up_blocks.append(UpBlock(in_channels=64*2,  out_channels=64 ))
        up_blocks.append(UpBlock(in_channels=32*2,  out_channels=32 ))
        
        self.up_blocks = nn.ModuleList(up_blocks)

        self.se  = SE_Block(c=32,r=4)

        self.out = nn.Conv2d(32, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x, istrain=False):

        # stages = dict()
        # stages[f"layer_0"] = x

        # # encoder
        # for i, block in enumerate(self.down_blocks, 1):
        #     x = block(x)
        #     stages[f"layer_{i}"] = x

        # # boundary enhancement moudel(BEM)
        # stage1 = stages[f"layer_1"]
    
        features = x[0]
        B_out  = x[1]

        x = self.bridge(features[-1])

        # decoder
        for i, block in enumerate(self.up_blocks, 1):
            key = self.depth + 1 - i
            x   = block(x, features[self.depth + 1 - i])

        x = self.se(x)
        x = self.out(x)
        del features

        if self.training:
            return x, B_out
        else:
            return x    
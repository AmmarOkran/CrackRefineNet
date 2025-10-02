import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = nn.Tanh()
        super().__init__(conv2d, upsampling)
        # super().__init__(conv2d, upsampling, activation)


class ConvBlock(nn.Module):

    def __init__(self, features_in, features_out) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=features_in, out_channels=features_out, kernel_size=3, padding='same')
        self.norm = nn.InstanceNorm2d(features_out)
    
    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)

        return x

class PAB(nn.Module):
    def __init__(self, in_channels, out_channels, pab_channels=64):
        super(PAB, self).__init__()
        # Series of 1x1 conv to generate attention feature maps
        self.pab_channels = pab_channels
        self.in_channels = in_channels
        self.top_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.center_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)

        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h * w, h * w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        return x


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.layer1 = ConvBlock(pyramid_channels, pyramid_channels)
        self.layer2 = ConvBlock(pyramid_channels, pyramid_channels)        

        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.conv_gate = nn.Conv2d(in_channels=pyramid_channels*2, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x1 = torch.cat((skip, x), dim=1)
        logits = self.conv_gate(x1)
        weights = torch.sigmoid(logits)
        x3 = skip * weights # b x f x w x h
        x2 = x3 + x

        x = self.layer1(x2)
        x = self.layer2(x)

        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


@HEADS.register_module()
class FPNDecoderTa(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        pab_channels=64,
        upsampling=4,
        use_center = False,
        **kwargs
    ):
        super(FPNDecoderTa, self).__init__(**kwargs)
        self.use_center = use_center
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        
        if self.use_center:
            self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)

        self.segmentation_head = SegmentationHead(
            in_channels=self.out_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, *features):
        c2, c3, c4, c5 = features[0][-4:]
        
        if self.use_center:
            neck = self.center(c5)
            p5 = self.p5(neck)
        else:
            p5 = self.p5(c5)
        
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)

        #if self.use_center:
        #    neck = F.interpolate(self.center(c5), scale_factor=8, mode="bilinear", align_corners=True)
        #    feature_pyramid.insert(0, neck)
        #    x = self.merge(feature_pyramid)

        x = self.dropout(x)

        x = self.segmentation_head(x)

        return x

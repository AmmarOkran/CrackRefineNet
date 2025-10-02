import torch
from torch import nn
from torch.nn import functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import EncoderMixin
from ..utils.non_local import NLBlockND, ModNLBlockND

__all__ = ["DeepLabV3Decoder"]



class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=2):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = Activation(activation)
        super().__init__(conv2d, upsampling)


@HEADS.register_module()
class DeepLabV3HeadNonLocalAlf(BaseDecodeHead):
    def __init__(self, 
                 out_channels=256, 
                 atrous_rates=(12, 24, 36), 
                 upsampling=8,
                 **kwargs):

        super(DeepLabV3HeadNonLocalAlf, self).__init__(**kwargs)
        
        self.in_channels = self.in_channels
        self.out_channels = out_channels
        self.num_classes = self.num_classes

        self.aspp = nn.Sequential(ASPP(self.in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
        self.segmentation_head = SegmentationHead(
            in_channels=out_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            upsampling=upsampling,
        )
    
    def forward(self, *features):
        x = self.aspp(features[0][-1])
        x = self.segmentation_head(x)
        return x



class LFEMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, stride=1):
        super(LFEMBlock, self).__init__()

        # Dilated convolutional layer
        self.conv_dilated = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation, stride=stride
        )

        # 1x1 convolutional layer
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Pass through dilated convolutional layer
        x = self.conv_dilated(x)

        # Pass through 1x1 convolutional layer
        out = self.conv_1x1(x)

        return x, out

class LocalFeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(LocalFeatureEnhancementModule, self).__init__()

        # Block 1
        self.block1 = LFEMBlock(in_channels, out_channels=32, dilation=2, kernel_size=3)

        # Block 2
        self.block2 = LFEMBlock(32, out_channels=64, dilation=4, kernel_size=3, stride=2)

        # Block 3
        self.block3 = LFEMBlock(64, out_channels=128, dilation=3, kernel_size=3, stride=2)


    def forward(self, x):
        # Block 1
        x_block1, out1 = self.block1(x)

        # Block 2 (input is the output of Block 1)
        x_block2, out2 = self.block2(x_block1)

        # Block 3 (input is the output of Block 2)
        x_block3, out3 = self.block3(x_block2)

        return [out1, out2, out3]
    


# class LFEM(nn.Sequential):
#     def __init__(self, in_channels=3, out_channels=None, dilation=None, kernel=None):
#         super().__init__()
#         modules = []
#         for i in range(len(out_channels)):
#             if i != 2:
#                 j = i if i != 3 else i - 1
#                 modules.append(nn.Sequential(nn.AdaptiveAvgPool2d(32),
#                                              nn.Conv2d(in_channels, out_channels[i], kernel_size=3, padding=0, dilation=dilation[j]),
#                                              nn.Conv2d(out_channels[i], out_channels[i], 1, padding=0)))
#                 in_channels = out_channels[i]
#         self.convs = nn.ModuleList(modules)
        
#     def forward(self, input):
#         out = []
#         for i, conv in enumerate(self.convs):
#             input = conv[0](input)
#             input = conv[1](input)
#             out.append(conv[2](input))
#         return out


        

@HEADS.register_module()
class DeepLabV3PlusHeadNonLocalAlf(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        out_decode=256,
        upsampling=8,
        n_blocks=4,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        bn_layer=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))
     
        # self.out_channels = out_channels if not out_channels is None else 2
        channels = encoder_channels[::-1]
        self.out_decode = out_decode
        self.output_stride = output_stride
        at_rates=(1, 2, 3)
        # self.non_local = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
        # self.blocks = nn.ModuleList(
        #     [NLBlockND(in_channels=channels[i], mode='concatenate', dimension=2, bn_layer=bn_layer) for i in range(n_blocks)]
        # )
        self.lfem = LocalFeatureEnhancementModule(3)
        # nn.ModuleList(
        #     [LFEM(channels[i], channels[i+1], at_rates[i]) for i in range(n_blocks)]
        # )
        self.blocks = nn.ModuleList(
            [ModNLBlockND(in_channels=channels[i], mode='concatenate', dimension=2, bn_layer=bn_layer) for i in range(n_blocks)]
        )
        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_decode, atrous_rates, separable=True),
            SeparableConv2d(out_decode, out_decode, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )

        # self.aspp2 = nn.Sequential(
        #     ASPP(encoder_channels[-1], out_decode, atrous_rates[1], separable=True),
        #     SeparableConv2d(out_decode, out_decode, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_decode),
        #     nn.ReLU(),
        # )
        self.up_samp = nn.UpsamplingBilinear2d(scale_factor=4)

        scale_factor = 2 if output_stride == 8 else 8 #4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_decode + channels[1], #  + channels[1] + channels[3]
                out_decode,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )
        self.segmentation_head = SegmentationHead(
            in_channels=out_decode,
            out_channels=self.out_channels,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, *features):
        
        img = features[0][0]
        features = features[0][1:]
        features = features[::-1]
        out = self.lfem(img)
        att_features = [self.blocks[i](features[i]) for i in range(len(self.blocks))]
        up_att_features = [self.up_samp(att_features[i]) for i in range(len(att_features))]
        # aspp_features = self.aspp(att_features[0])
        aspp_features = self.aspp(features[0])
        aspp_features = self.up(aspp_features)
        # high_res_features = self.block1(att_features[-1])
        high_res_features = self.block1(features[-1])
        concat_features = torch.cat([up_att_features[1], aspp_features, high_res_features], dim=1)
        # concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        # concat_features = torch.cat([att_features[1], out[2], aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return self.segmentation_head(fused_features)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)

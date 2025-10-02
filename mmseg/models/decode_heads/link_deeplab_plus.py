import torch
from torch import nn
from torch.nn import functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from ..utils import modules as md



class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=2):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = Activation(activation)
        super().__init__(conv2d, upsampling)


class TransposeX2(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))

        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            md.Conv2dReLU(
                in_channels,
                in_channels // 4,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            ),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            md.Conv2dReLU(
                in_channels // 4,
                out_channels,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            ),
        )

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x
    

class LinknetDecoder(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        n_blocks=5,
        use_batchnorm=True,
        upsampling=1,
        **kwargs
    ):
        super(LinknetDecoder, self).__init__(**kwargs)

        # remove first skip
        # encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList(
            [DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm) for i in range(n_blocks)]
        )

        # self.segmentation_head = SegmentationHead(
        #     in_channels=32, out_channels=self.num_classes, kernel_size=1, upsampling=upsampling
        # )

    def forward(self, *features):
        features = features[0][:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]
        link_res = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            link_res.append(x)
        # x = self.segmentation_head(x)

        return link_res
    


class DeepLabV3Head(BaseDecodeHead):
    def __init__(self, 
                 out_channels=256, 
                 atrous_rates=(12, 24, 36), 
                 upsampling=8,
                 **kwargs):

        super(DeepLabV3Head, self).__init__(**kwargs)
        
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

    # def forward(self, *features):
    #     return super().forward(features[-1])



class DeepLabV3PlusHead(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        out_decode=256,
        upsampling=8,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))
     
        # self.out_channels = out_channels if not out_channels is None else 2
        self.out_decode = out_decode
        self.output_stride = output_stride

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
                highres_out_channels + out_decode,
                out_decode//2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_decode//2),
            nn.ReLU(),
        )
        self.segmentation_head = SegmentationHead(
            in_channels=out_decode,
            out_channels=self.out_channels,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[0][-1])
        # aspp_features2 = self.aspp2(features[0][-1])
        aspp_features = self.up(aspp_features)
        # aspp_features2 = self.up(aspp_features2)
        # aspp_features += aspp_features2
        high_res_features = self.block1(features[0][-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
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



@HEADS.register_module()
class DeepLabV3PlusLinkHeadUpdated(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        n_blocks=5,
        dropout=0.2,
        use_batchnorm=True,
        out_decode=256,
        upsampling=8,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.link = LinknetDecoder(list(encoder_channels[:-1]) + [1280], n_blocks=n_blocks, **kwargs)
        self.deeplab = DeepLabV3PlusHead(encoder_channels, **kwargs)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.b_fuse = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=1,
            upsampling=2,
        )

    
    def forward(self, *features):

        features = features[0][:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = list(features[1:])

        # Deeplabv3++
        aspp_features_1 = self.deeplab.aspp(x)
        aspp_features_2 = self.deeplab.up(aspp_features_1)
        high_res_features = self.deeplab.block1(skips[-1])
        concat_features = torch.cat([aspp_features_2, high_res_features], dim=1)
        fused_features = self.deeplab.block2(concat_features)

        # LinkNet
        skips[2] = fused_features + skips[2]
        x = torch.cat([x, aspp_features_1], dim=1)
        for i, decoder_block in enumerate(self.link.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        x1 = self.up(self.b_fuse(fused_features))
        # x = x + x1
        x = torch.cat([x, x1], dim=1)
        x = self.dropout(x)
        return self.segmentation_head(x)
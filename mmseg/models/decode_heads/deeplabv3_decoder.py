import torch
from torch import nn
from torch.nn import functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import EncoderMixin

__all__ = ["DeepLabV3Decoder"]



class SEAttention(nn.Module):
    """Squeeze-and-Excitation (SE) attention block."""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: fully connected layers and sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DecoderFusionBlock(nn.Module):
    """
    Fuses decoder features and heatmap predictions via concatenation,
    convolution, and attention.
    """
    def __init__(self, decoder_channels, heatmap_channels, out_channels, reduction=16):
        super(DecoderFusionBlock, self).__init__()
        # After concatenation, total channels = decoder_channels + heatmap_channels
        self.conv = nn.Conv2d(decoder_channels + heatmap_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SEAttention(out_channels, reduction)
        
        # If you wish to add a residual connection, ensure input/output dims match.
        self.residual_conv = None
        if decoder_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            # Project decoder feature to out_channels
            self.residual_conv = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)
    
    def forward(self, decoder_feat, heatmap):
        # Ensure heatmap and decoder features have the same spatial dimensions
        if heatmap.shape[2:] != decoder_feat.shape[2:]:
            heatmap = F.interpolate(heatmap, size=decoder_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate along the channel dimension
        fused = torch.cat([decoder_feat, heatmap], dim=1)
        fused = self.conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = self.attention(fused)
        
        # Optionally, add a residual connection from the decoder feature (after matching dimensions)
        res = self.residual_conv(decoder_feat)
        fused = fused + res
        
        return fused
    

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=2):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = Activation(activation)
        super().__init__(conv2d, upsampling)



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

class DecoderBlock(nn.Module):

    def __init__(self, features_in, features_out) -> None:
        super().__init__()

        self.dconv = nn.ConvTranspose2d(in_channels=features_in, out_channels=features_out, kernel_size=2, stride=2)

        self.layer1 = ConvBlock(features_out, features_out)
        self.layer2 = ConvBlock(features_out, features_out)        

        self.conv_gate = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, bias=False)
    
    def forward(self, x, skip=None):
        #x: b x 2f x w/2 x h/2
        x1 = self.dconv(x) # b x f x w x 
        x2 = torch.cat((skip, x1), dim=1) # b x 2f x w x h
        
        logits = self.conv_gate(x2)
        weights = torch.sigmoid(logits) # b x 1 x w x h
        x3 = skip * weights # b x f x w x h

        x = x3 + x1

        # print("Sum: ", x.shape)

        x = self.layer1(x)
        x = self.layer2(x)

        return x

@HEADS.register_module()
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


@HEADS.register_module()
class DeepLabV3PlusHead(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        out_decode=256,
        upsampling=8,
        atrous_rates=(12, 24, 36),
        output_stride=16,
        fusion = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))
     
        # self.out_channels = out_channels if not out_channels is None else 2
        self.out_decode = out_decode
        self.output_stride = output_stride
        self.fusion = fusion
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

        scale_factor = 2 if output_stride == 8 else 8 #4 8
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        # self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.att = DecoderBlock(out_decode, out_decode*2)
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
                out_decode,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(32, highres_out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(highres_out_channels),
        #     nn.ReLU(),
        # )
        # self.block3 = nn.Sequential(
        #     SeparableConv2d(
        #         32,
        #         2,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        # )

        if self.fusion:
            self.fusion_block = DecoderFusionBlock(precise_channels=out_decode, 
                                               heatmap_channels=1, 
                                               fusion_out_channels=out_decode)
        self.segmentation_head = SegmentationHead(
            in_channels=out_decode,
            out_channels=self.out_channels,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, *features, heatmap=None):
        aspp_features = self.aspp(features[0][-1])
        # aspp_features2 = self.aspp2(features[0][-1])
        # aspp_features = self.att(aspp_features, features[0][-2])
        aspp_features = self.up(aspp_features)
        # aspp_features2 = self.up(aspp_features2)
        # aspp_features += aspp_features2
        # eff_high_res = self.block3(self.max_pool_layer(features[0][0]))
        high_res_features = self.block1(features[0][-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        # z = self.up2(self.block3(features[0][0]))
        if self.fusion:
            fused_features = self.fusion_block(fused_features, heatmap)
            return self.segmentation_head(fused_features)
        return self.segmentation_head(fused_features)#+z


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

import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from ..utils import modules


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
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
            modules.Conv2dReLU(
                in_channels,
                in_channels // 4,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            ),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            modules.Conv2dReLU(
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

@HEADS.register_module()
class LinknetDecoderRes(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        n_blocks=5,
        use_batchnorm=True,
        upsampling=1,
        **kwargs
    ):
        super(LinknetDecoderRes, self).__init__(**kwargs)

        # remove first skip
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList(
            [DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm) for i in range(n_blocks)]
        )

        self.segmentation_head = SegmentationHead(
            in_channels=64, out_channels=self.num_classes, kernel_size=1, upsampling=upsampling
        )

    def forward(self, *features):
        features = features[0]#[1:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        x = self.segmentation_head(x)

        return x

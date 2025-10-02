import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

import matplotlib.pyplot as plt


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = nn.Tanh()
        super().__init__(conv2d, upsampling)
        # super().__init__(conv2d, upsampling, activation)

@HEADS.register_module()
class FusionHead(BaseDecodeHead):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=64,
        dropout=0.2,
        merge_policy="add",
        pab_channels=64,
        upsampling=4,
        use_center = False,
        **kwargs
    ):
        super(FusionHead, self).__init__(**kwargs)

        self.side = nn.ModuleList() 
        for i in range(encoder_depth):
            if i < 3:
                self.side.append(nn.Conv2d(segmentation_channels * (2**i), 2, (1,1)))
            else:
                self.side.append(nn.Conv2d(segmentation_channels * 8, 2, (1,1)))
    
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.segmentation_head = SegmentationHead(
            in_channels=2,
            out_channels=self.num_classes,
            kernel_size=1,
            # upsampling=upsampling,
        )

    def forward(self, *features):

        features = features[0]#[:-1]
        f = []
        # fig, axarr = plt.subplots(5)
        for i, fetaure in enumerate(features):
            f.append(F.interpolate(self.side[i](fetaure), scale_factor=2**(2+i), mode="nearest"))
            # axarr[i].imshow(f[i].squeeze()[0].detach().cpu().numpy())
        x = sum(f)
        x = self.dropout(x)

        # x = self.segmentation_head(x)

        return x

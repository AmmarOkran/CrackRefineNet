from typing import List, Union
import timm
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class TimNets(BaseModule):
    def __init__(self, 
                 name, 
                 pret=True, 
                 in_channels=3, 
                 depth=5, 
                 output_stride=32,
                 **kwargs):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pret,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")
        
        if name.startswith("tu-"):
            name = name[3:]

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
    

# @MODELS.register_module()
# class TimHRNet(BaseModule):
#     def __init__(self, 
#                  name, 
#                  in_channels=3, 
#                  depth=5, 
#                  weights=None, 
#                  output_stride=32, 
#                  **kwargs):
#         super().__init__()
#         if name.startswith("tu-"):
#             name = name[3:]
#             self.encoder = TimmUniversalEncoder(
#                 name=name,
#                 in_channels=in_channels,
#                 depth=depth,
#                 output_stride=output_stride,
#                 pret=weights is not None,
#             )
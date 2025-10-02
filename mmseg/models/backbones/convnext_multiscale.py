import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from mmpretrain.models.backbones import ConvNeXt
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from functools import partial
from mmpretrain.models.utils import GRN, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import is_model_wrapper
import copy
from itertools import chain
from mmengine.dist import broadcast
import torchvision.models as models
import matplotlib.pyplot as plt

class VGGStage1(nn.Module):
    def __init__(self, num_output_channels, output_resolution):
        super(VGGStage1, self).__init__()
        # Load pre-trained weights for the modified layer from the original VGG16 model
        original_conv1_weights = models.vgg16(pretrained=True).features[0:4]
        self.conv1 = original_conv1_weights[0]
        self.relu1 = original_conv1_weights[1]
        self.conv2 = original_conv1_weights[2]
        self.relu2 = original_conv1_weights[3]
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
    

@MODELS.register_module()
class ConvNeXtMultiScale(ConvNeXt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Downsampling scales
        self._init_model_weights()

        self.scale_img = nn.AvgPool2d(kernel_size=2)

        # List of different output channels and resolutions for each variant
        output_channels_list = [128, 256, 512, 1024]
        output_resolutions_list = [(32, 32), (16, 16), (8, 8), (4, 4)]

        # Loop through each variant
        self.vgg = nn.ModuleList()
        # for i, (num_output_channels, output_resolution) in enumerate(zip(output_channels_list, output_resolutions_list)):
        #     # Create a new model instance for each variant
        #     vgg_stage1 = VGGStage1(num_output_channels, output_resolution)

        #     # modified_conv1_weights = torch.zeros(num_output_channels, 3, 3, 3)
        #     # modified_conv1_weights.copy_(original_conv1_weights[:, :3])
        #     # modified_conv1_weights[:, :3, :, :] = original_conv1_weights[:num_output_channels, :, :, :]
        #     # vgg_stage1.conv1.weight.data = modified_conv1_weights
        #     self.vgg.append(vgg_stage1)

        #  Assuming in_channels is the number of channels in the original feature map
        # self.conv1x1 = nn.ModuleList()
        # for i, stage in enumerate(self.stages):
        #     # self.conv1x1.append(nn.Conv2d(stage[0].depthwise_conv.in_channels+3, stage[0].depthwise_conv.in_channels, kernel_size=1, stride=1, padding=0).cuda())
        #     s1 = nn.Conv2d(3, stage[0].depthwise_conv.out_channels, stage[0].depthwise_conv.kernel_size, 
        #                    stride=stage[0].depthwise_conv.stride, padding=stage[0].depthwise_conv.padding)
        #     with torch.no_grad():
        #         # s1.weight[:, :stage[0].depthwise_conv.in_channels].copy_(stage[0].depthwise_conv.weight)
        #         # s1.weight[:, stage[0].depthwise_conv.in_channels:].copy_(stage[0].depthwise_conv.weight[:, :3])
        #         s1.weight.copy_(stage[0].depthwise_conv.weight[:, :3])
        #         s1.bias.copy_(stage[0].depthwise_conv.bias)
        #         # s1.bias[stage[0].depthwise_conv.in_channels:].copy_(stage[0].depthwise_conv.bias[:3])
        #     # self.stages[i][0].depthwise_conv = s1
        #     self.conv1x1.append(nn.Sequential(s1,
        #                                       nn.BatchNorm2d(stage[0].depthwise_conv.out_channels),
        #                                       nn.ReLU(),))
        
        
    def _init_model_weights(self) -> None:
        """Initialize the model weights if the model has
        :meth:`init_weights`"""
        model = self.model.module if is_model_wrapper(
            self) else self
        if hasattr(model, 'init_weights'):
            model.init_weights()
            # sync params and buffers
            for name, params in model.state_dict().items():
                broadcast(params)

    def forward(self, x):
        scale_img_2 = self.scale_img(self.scale_img(x))
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)
        scale_img_5 = self.scale_img(scale_img_4)
        scale_imgs = [scale_img_2, scale_img_3, 
                      scale_img_4, scale_img_5]

        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            # concatenated_feature = torch.cat([x, scale_imgs[i]], dim=1) 
            # x += self.conv1x1[i](scale_imgs[i])
            # x[:, :self.vgg[i](scale_imgs[i]).shape[1]] += self.vgg[i](scale_imgs[i])
            # x[:, :3, :, :] *= scale_imgs[i]
            # if i != 3:
            #     x[:, -3:, :, :] += scale_imgs[i]
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x)+x)

        return tuple(outs)
    
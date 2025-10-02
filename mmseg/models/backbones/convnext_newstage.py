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

from mmpretrain import get_model


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x



@MODELS.register_module()
class ConvNeXtModified(ConvNeXt):
    def __init__(self,norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                #  frozen_stages=None,
                 layer_scale_init_value=1e-6,
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        # self.frozen_stages = frozen_stages
        self._init_model_weights()

        self.eff = EffModel()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        # self.drop_path_rate = kwargs['drop_path_rate']
        # original_module_list = self.downsample_layers
        # # Modify the ModuleList
        # modified_module_list = nn.ModuleList([
        #     nn.Sequential(
        #         # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #         nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        #         build_norm_layer(norm_cfg, 32),
        #         # nn.UpsamplingBilinear2d(scale_factor=2)
        #     ),
        #     nn.Sequential(
        #         build_norm_layer(norm_cfg, 32),
        #         nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(4, 4)),
        #     ),
        #     *original_module_list[1:]  # Keep the rest of the original modules
        # ])

        # original_weights = original_module_list[0][0].weight.clone()
        # original_biass = original_module_list[0][0].bias.clone()
        # # Take a subset of the original weights (16 out of 128)
        # subset_weights = original_weights[:32]
        # subset_bias = original_biass[:32]

        # # Resize the spatial dimensions to (3, 3)
        # resized_weights = F.interpolate(subset_weights, size=(5, 5), mode='nearest')# Expand the number of channels from 3 to 16 using a linear transformation
        
        # # # Reshape the weights to (128, 48) and the bias to (128,)
        # # reshaped_weights = original_weights.view(128, -1)
        # # expanded_weights = F.linear(reshaped_weights, weight=torch.randn((16 * 4 * 4, 48))).view(128, 32, 4, 4)



        # # Copy weights from the first modified Sequential to the second modified Sequential
        # with torch.no_grad():
        #     modified_module_list[0][0].weight.copy_(resized_weights)
        #     modified_module_list[0][1].bias.copy_(subset_bias)
        #     modified_module_list[1][1].weight[:, :3] = original_weights
        #     modified_module_list[1][1].weight[:, 3:6] = original_weights
        #     modified_module_list[1][1].weight[:, 6:9] = original_weights
        #     modified_module_list[1][1].weight[:, 9:12] = original_weights
        #     modified_module_list[1][1].weight[:, 12:15] = original_weights
        #     # modified_module_list[1][1].weight[:, 15] = original_weights[:, 0]
        #     modified_module_list[1][1].weight[:, 15:18] = original_weights
        #     modified_module_list[1][1].weight[:, 18:21] = original_weights
        #     modified_module_list[1][1].weight[:, 21:24] = original_weights
        #     modified_module_list[1][1].weight[:, 24:27] = original_weights
        #     modified_module_list[1][1].weight[:, 27:30] = original_weights
        #     modified_module_list[1][1].weight[:, 30:] = original_weights[:, :2]
        #     # modified_module_list[1][1].weight.copy_(expanded_weights)
        #     modified_module_list[1][0].bias.copy_(subset_bias)
        
        # # Display the modified ModuleList
        # # print(modified_module_list)
        # # print()
        # self.downsample_layers = modified_module_list
        # # stochastic depth decay rule
        # dpr = [
        #     x.item()
        #     for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        # ]
        # block_idx = 0
        # # self.stages[0][0].depthwise_conv = nn.Conv2d(128, 128, 7, 1, padding=3, groups=128)
        # stgs = nn.ModuleList([*self.stages])#[1:]
        # chnls = [32, 128, 256, 512, 1024]
        # for i in range(1):
        #     depth = 3#self.depths[i]
        #     channels = chnls[i]

        #     # if i >= 1:
        #     stage = nn.Sequential(*[
        #         ConvNeXtBlock(
        #             in_channels=channels,
        #             # dw_conv_cfg=dict(kernel_size=3, padding=1),
        #             drop_path_rate=dpr[block_idx + j],
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #             linear_pw_conv=linear_pw_conv,
        #             layer_scale_init_value=layer_scale_init_value,
        #             use_grn=use_grn,
        #             with_cp=with_cp) for j in range(depth)
        #             ])
        #     block_idx += depth

        #     stgs.insert(i, stage)
        # self.out_indices.insert(4, 4)
        # self.stages = stgs

        # with torch.no_grad():
        #     # Depthwise Convolution
        #     # depthwise_weights = F.interpolate(self.stages[1][0].depthwise_conv.weight[:32, :, :, :], size=(3, 3), mode='bilinear', align_corners=False)
        #     depthwise_weights = self.stages[1][0].depthwise_conv.weight[:32, :, :, :]
        #     self.stages[0][0].depthwise_conv.weight.copy_(depthwise_weights)
        #     self.stages[0][0].depthwise_conv.bias.copy_(self.stages[1][0].depthwise_conv.bias[:32])

        #     self.stages[0][1].depthwise_conv.weight.copy_(depthwise_weights)
        #     self.stages[0][1].depthwise_conv.bias.copy_(self.stages[1][0].depthwise_conv.bias[:32])

        #     self.stages[0][2].depthwise_conv.weight.copy_(depthwise_weights)
        #     self.stages[0][2].depthwise_conv.bias.copy_(self.stages[1][0].depthwise_conv.bias[:32])
        #     # LayerNorm
        #     self.stages[0][0].norm.weight.copy_(self.stages[1][0].norm.weight[:32])
        #     self.stages[0][0].norm.bias.copy_(self.stages[1][0].norm.bias[:32])

        #     self.stages[0][1].norm.weight.copy_(self.stages[1][0].norm.weight[:32])
        #     self.stages[0][1].norm.bias.copy_(self.stages[1][0].norm.bias[:32])

        #     self.stages[0][2].norm.weight.copy_(self.stages[1][0].norm.weight[:32])
        #     self.stages[0][2].norm.bias.copy_(self.stages[1][0].norm.bias[:32])

        #     # Pointwise Conv1
        #     resh_weight = self.stages[1][0].pointwise_conv1.weight.view(1, 1, 512, 128)
        #     # Use F.interpolate to resize the tensor to [1, 1, 64, 16]
        #     adapted_weights = F.interpolate(resh_weight, size=(128, 32), mode='bilinear', align_corners=False)
        #     # Reshape back to [64, 16]
        #     final_adapted_weights = adapted_weights.view(128, 32)
        #     self.stages[0][0].pointwise_conv1.weight.copy_(final_adapted_weights)
        #     self.stages[0][0].pointwise_conv1.bias.copy_(self.stages[1][0].pointwise_conv1.bias[:128])

        #     self.stages[0][1].pointwise_conv1.weight.copy_(final_adapted_weights)
        #     self.stages[0][1].pointwise_conv1.bias.copy_(self.stages[1][0].pointwise_conv1.bias[:128])

        #     self.stages[0][2].pointwise_conv1.weight.copy_(final_adapted_weights)
        #     self.stages[0][2].pointwise_conv1.bias.copy_(self.stages[1][0].pointwise_conv1.bias[:128])

        #     # Pointwise Conv2
        #     resh_weight = self.stages[1][0].pointwise_conv2.weight.view(1, 1, 128, 512)
        #     # Use F.interpolate to resize the tensor to [1, 1, 64, 16]
        #     adapted_weights = F.interpolate(resh_weight, size=(32, 128), mode='bilinear', align_corners=False)
        #     # Reshape back to [64, 16]
        #     final_adapted_weights = adapted_weights.view(32, 128)
        #     self.stages[0][0].pointwise_conv2.weight.copy_(final_adapted_weights)
        #     self.stages[0][0].pointwise_conv2.bias.copy_(self.stages[1][0].pointwise_conv2.bias[:32])

        #     self.stages[0][1].pointwise_conv2.weight.copy_(final_adapted_weights)
        #     self.stages[0][1].pointwise_conv2.bias.copy_(self.stages[1][0].pointwise_conv2.bias[:32])

        #     self.stages[0][2].pointwise_conv2.weight.copy_(final_adapted_weights)
        #     self.stages[0][2].pointwise_conv2.bias.copy_(self.stages[1][0].pointwise_conv2.bias[:32])

        # for i in reversed(range(len(chnls))):
        #     if i in self.out_indices:
        #         if i > 1:
        #             self._modules[f'norm{i}'] = self._modules.pop(f'norm{i-1}')
        #         else:
        #             norm_layer = build_norm_layer(norm_cfg, chnls[i])
        #             self.add_module(f'norm{i}', norm_layer)

                # self._modules['custom_norm'] = self._modules.pop('norm0')
                # if f'norm{i}' in self._modules:


                # self._modules['custom_norm'] = model._modules.pop('norm0')
                # norm_layer = build_norm_layer(norm_cfg, chnls[i])
                # self.add_module(f'norm{i}', norm_layer)
        # self._freeze_stages()
    
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
        outs = []
        eff_0_1 = self.eff(x)
        eff_up = self.up(eff_0_1[0])
        # eff_up = eff_0_1[0]
        # eff_up = torch.mean(eff_up, dim=1).unsqueeze(1)
        x = torch.cat([x, eff_up], 1)
        outs.append(eff_0_1[0])
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)
    

    # def _freeze_stages(self):
    #     for i in range(len(self.stages)):
    #         if i >= self.frozen_stages and self.frozen_stages !=0:
    #             downsample_layer = self.downsample_layers[i]
    #             stage = self.stages[i]
    #             downsample_layer.eval()
    #             stage.eval()
    #             for param in chain(downsample_layer.parameters(),
    #                             stage.parameters()):
    #                 param.requires_grad = False




class EffModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k  efficientnet-b5_3rdparty-ra-noisystudent_in1k
        # efficientnet-b5_3rdparty_8xb32_in1k
        self.eff = get_model("efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k", pretrained=True, 
                             head=None, neck=None, backbone=dict(out_indices=(1,)))

    def forward(self, x):
        x = self.eff(x)
        return x
    

if __name__ == "__main__":

    img = torch.rand(1, 3, 128, 128)
    model = ConvNeXtModified()
    pred = model(img)
    print(pred[0].shape)


import torch
import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models.torchvision_models import pretrained_settings
# from ..utils._base import EncoderMixin

from mmengine.model import BaseModule
from mmseg.registry import MODELS

# from ..utils._base import replace_strides_with_dilation, patch_first_conv



def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)

    def get_stages(self):
        """Override it in your implementation"""
        raise NotImplementedError

    def make_dilated(self, output_stride):

        if output_stride == 16:
            stage_list = [
                1,
            ]
            dilation_list = [
                2,
            ]

        elif output_stride == 8:
            stage_list = [0, 1]
            dilation_list = [2, 4]

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))

        self._output_stride = output_stride

        stages = [self.layer3, self.layer4]
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


class ECB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 3), padding='same')

    
    def forward(self, x):
        k1 = self.conv1(x)
        k2 = self.conv2(x)
        k3 = self.conv3(x)
        x = k1 + k2 + k3

        return x



class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, depth=5, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def make_dilated(self, *args, **kwargs):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling" " operations for downsampling!")

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages


    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k, None)
        super().load_state_dict(state_dict, **kwargs)

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# fmt: on

vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}

@MODELS.register_module()
class VggEnhanced(BaseModule):
    def __init__(self, name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
        super(VggEnhanced, self).__init__()
        encoders = {}
        encoders.update(vgg_encoders)

        try:
            Encoder = encoders[name]["encoder"]
        except KeyError:
            raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))
        
        params = encoders[name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            try:
                settings = encoders[name]["pretrained_settings"][weights]
            except KeyError:
                raise KeyError(
                    "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                        weights,
                        name,
                        list(encoders[name]["pretrained_settings"].keys()),
                    )
                )
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))

        encoder.set_in_channels(in_channels, pretrained=weights is not None)
        if output_stride != 32:
            encoder.make_dilated(output_stride)
        self.encoder=encoder


    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.encoder.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages
    
    def chang_conv(self):

        stages = self.encoder.get_stages()
        for stage in stages:
            for name, layer in stage.named_children():
                if isinstance(layer, nn.Conv2d):
                    # Replace the convolutional layer with the custom layer
                    c_conv = stage[int(name)]
                    stage[int(name)] = ECB(layer.in_channels, layer.out_channels).cuda()
                    # Load pretrained weights from the original VGG model to the custom layer
                    # stage[int(name)].conv3.weight.data = c_conv.weight.data
                    # stage[int(name)].conv3.bias.data = c_conv.bias.data
        return stages
    

    def forward(self, x):
        # stages = self.encoder.get_stages()
        stages = self.chang_conv()
        features = []
        for i in range(self.encoder._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features
        
if __name__ == "__main__":
    model = VggEnhanced('vgg16', weights='imagenet')
    x = torch.rand(4, 3, 32, 32)
    print(model(x))
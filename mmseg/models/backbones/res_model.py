
import torch.nn as nn
import torchvision.models as models
from mmseg.registry import MODELS


@MODELS.register_module()
class Res50(nn.Module):
    def __init__(self, pretrained=True, depth=5, style='pytorch', contract_dilation=True):
        super(Res50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained, )
        self.model._depth = depth
        # self._out_channels = out_channels
        self.model._in_channels = 3
        del self.model.fc
        del self.model.avgpool
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu),
            nn.Sequential(self.model.maxpool, self.model.layer1),
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]


    def forward(self, x):  # should return a tuple
        stages = self.get_stages()

        features = []
        for i in range(self.model._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def init_weights(self, pretrained=None):
        pass

if __name__ == "__main__":
    model = Res50()
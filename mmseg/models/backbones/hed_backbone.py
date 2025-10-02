import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch.utils import model_zoo
from collections import OrderedDict



@MODELS.register_module()
class HEDBackbone(BaseModule):
    def __init__(self):
        super().__init__()

        self.vgg_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x):
        x = F.relu(self.vgg_conv1(x))
        x = F.relu(self.vgg_conv2(x))
        feat1 = x
        # a1 = self.side1(x)
        # s1 = F.sigmoid(a1)
        x = self.vgg_pool1(x)
        x = F.relu(self.vgg_conv3(x))
        x = F.relu(self.vgg_conv4(x))
        feat2 = x
        # a2 = self.side2(x)
        # s2 = F.sigmoid(a2)
        x = self.vgg_pool2(x)
        x = F.relu(self.vgg_conv5(x))
        x = F.relu(self.vgg_conv6(x))
        x = F.relu(self.vgg_conv7(x))
        feat3 = x
        # a3 = self.side3(x)
        # s3 = F.sigmoid(a3)
        x = self.vgg_pool3(x)
        x = F.relu(self.vgg_conv8(x))
        x = F.relu(self.vgg_conv9(x))
        x = F.relu(self.vgg_conv10(x))
        feat4 = x
        # a4 = self.side4(x)
        # s4 = F.sigmoid(a4)
        x = self.vgg_pool4(x)
        x = F.relu(self.vgg_conv11(x))
        x = F.relu(self.vgg_conv12(x))
        x = F.relu(self.vgg_conv13(x))
        feat5 = x
        # a5 = self.side5(x)
        # s5 = F.sigmoid(a5)
        # fuse = self.fuse(a1, a2, a3, a4, a5)
        return [feat1, feat2, feat3, feat4, feat5]

def get_vgg_weights():
    # Download the pre-trained VGG wegiths and load in HED model.
    # Names of weights are modified from the original.
    vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
    vgg_state_dict = model_zoo.load_url(vgg16_url, model_dir="./")
    idx = 0
    partial_vgg = OrderedDict()
    for k, v in vgg_state_dict.items():
        if k == "classifier.0.weight":
            break
        new_key = "vgg_conv" + str(int(idx/2)+1) + k[k.rfind("."):]
        partial_vgg[new_key] = v
        idx+=1
    return partial_vgg

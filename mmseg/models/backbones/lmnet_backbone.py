import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from .lmnet_blocks import SEM, PFM  # We'll move all blocks like SEM, PFM into a `lmnet_blocks.py` file


@MODELS.register_module()
class LMNetBackbone(BaseModule):
    def __init__(self, in_channels=3):
        super().__init__()

        self.prelayer = nn.Conv2d(in_channels,96,kernel_size=3,padding=1,bias=False)
        self.sem1 = SEM(96, reduction=24)
        
        self.PFM1 = PFM(96, 32, 64, 32, 32, [8,24])
        self.PFM2 = PFM(96, 32, 64, 32, 32, [8,24])
        
        self.PFM3 = PFM(96, 32, 96, 32, 32, [8,32], True)
        self.PFM4 = PFM(128, 32, 96, 32, 32, [8,32])
        self.PFM5 = PFM(128, 32, 96, 32, 32, [8,32])
        
        self.PFM6 = PFM(128, 64, 128, 64, 64, [16,48], True)
        self.PFM7 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM8 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM9 = PFM(192, 64, 128, 64, 64, [16,48])
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=1,dilation=1,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=2,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3,padding=4,dilation=4,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 64)
            )
        self.sem2 = SEM(320, reduction=80)

        self.Max_Pooling = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.1)

    def forward(self, x):
        img_shape = x.shape[2:]
        x1 = self.prelayer(x)
        x2 = self.sem1(x1)
        x = x1+x2
        
        # encoding path
        i1 = self.PFM1(x)
        i2 = self.PFM2(i1)
        x = self.dropout1(self.Max_Pooling(i2))
        
        i3 = self.PFM3(x, True)
        i4 = self.PFM4(i3)
        i5 = self.PFM5(i4)
        x = self.dropout2(self.Max_Pooling(i5))

        i6 = self.PFM6(x, True)
        i7 = self.PFM7(i6)
        i8 = self.PFM8(i7)
        i9 = self.PFM9(i8)
        x = self.dropout3(self.Max_Pooling(i9))

        #D-Conv
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x1 = torch.cat([x1,x2,x3],1)
        x2 = self.sem2(x1)

        return [img_shape, i1, i2, i3, i4, i5, i6, i7, i8, i9, x, x1, x2]  # needed for FRCM + decoder
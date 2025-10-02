import torch
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import torch.nn as nn
import torch.nn.functional as F
from .mobileVitblock import MobileViTBlock
from mmseg.registry import MODELS
from mmengine.model import BaseModule


def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    
    return laplacekernel.float()


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=None):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)

class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, reduction):
        super(EEM, self).__init__()
        
        self.groups = groups
        self.gk = gaussiankernel(ch_in, int(ch_in/groups), kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, int(ch_in/groups), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )

        self.conv3 = nn.Sequential(
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Conv2d(int(ch_out/2), ch_out, kernel_size=1,padding=0,groups=2),
                    nn.PReLU(num_parameters=ch_out, init=0.01),
                    nn.GroupNorm(4, ch_out)
                    )
            
        self.sem1 = SEM(ch_out, reduction=reduction)  # let's put reduction as 2
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)
      
    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to('cuda'), padding='same',groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to('cuda'), padding='same',groups=self.groups)
        DoG = self.conv1(DoG-x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG*LoG)
        
        tot1 = self.sem1(tot)
        x1 = self.sem2(x)
        
        return self.prelu(x+x1+tot+tot1)


class SubSpace(nn.Module):
    """
    Subspace class.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out


class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    nout : int
        number of output feature maps

    h : int
        height of a input feature map

    w : int
        width of a input feature map

    num_splits : int
        number of subspaces

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out


class EfficientCrackNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.block1 = nn.Sequential(
            SeparableConv2d(3, 16, 3),
            nn.BatchNorm2d(16),
        )

        self.block2 = nn.Sequential(
            SeparableConv2d(16, 32, 3),
            nn.BatchNorm2d(32),
        )

        self.block3 = nn.Sequential(
            SeparableConv2d(32, 64, 3),
            nn.BatchNorm2d(64),
        )

        self.block4 = nn.Sequential(
            SeparableConv2d(64, 16, 3),
            nn.BatchNorm2d(16),
        )

        self.block5 = nn.Sequential(
            SeparableConv2d(16, 32, 3),
            nn.BatchNorm2d(32),
        )

        self.block6 = nn.Sequential(
            SeparableConv2d(32, 64, 3),
            nn.BatchNorm2d(64),
        )

        self.EEM1 = EEM(ch_in=16, ch_out=16, kernel=3, groups=1, reduction=2)
        self.EEM2 = EEM(ch_in=32, ch_out=32, kernel=3, groups=1, reduction=2)
        self.EEM3 = EEM(ch_in=64, ch_out=64, kernel=3, groups=1, reduction=2)
        self.ULSAM1 = ULSAM(64, 64, 14, 28, 4)

        self.mv2_block1 = MobileViTBlock(dim=32, depth=3, channel=16, kernel_size=3, patch_size=(2, 2), mlp_dim=64)
        self.mv2_block2 = MobileViTBlock(dim=64, depth=3, channel=32, kernel_size=3, patch_size=(2, 2), mlp_dim=128)

        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.final_ulsam = ULSAM(32, 32, 24, 34, 4)
        self.final_conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        # Encoder Block 1
        enc1 = F.relu(self.block1(x))
        enc1 = self.EEM1(enc1)

        # Encoder Block 2
        enc2 = F.relu(self.block2(enc1))
        enc2 = self.EEM2(enc2)

        # Encoder Block 3
        enc3 = self.maxpool(F.relu(self.block3(enc2)))
        enc3 = self.EEM3(enc3)
        enc3 = self.ULSAM1(enc3)

        # Encoder Block 4
        enc4 = self.maxpool(F.relu(self.block4(enc3)))
        enc4 = self.mv2_block1(enc4)

        # Encoder Block 5
        enc5 = F.relu(self.block5(enc4))
        enc5 = self.mv2_block2(enc5)

        # Encoder Block 6
        enc6 = self.maxpool(F.relu(self.block6(enc5)))

        # Output to be reversed
        encoder_output = self.final_conv2(self.final_ulsam(self.final_conv1(enc6)))

        return [enc1, enc2, enc3, enc4, enc5, enc6, encoder_output]


@MODELS.register_module()
class EfficientCrackNet(BaseModule):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientCrackNetEncoder()

    def forward(self, x):
        return self.encoder(x)

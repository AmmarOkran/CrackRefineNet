import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


class PDAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction, dropout):
        super(PDAM, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv1b = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv2a = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv2b = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv3a = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=4,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out,ch_out,kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
            )
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout2d(dropout)
        self.sem = SEM(ch_in[0], reduction=reduction)
        
    def forward(self,x,x1,x2):
        x0 = self.sem(x)
        x0 = torch.cat([self.conv1a(x+x0),self.conv1b(x+x0)],1)
        x1 = torch.cat([self.conv2a(x1),self.conv2b(x1)],1)
        x2 = torch.cat([self.conv3a(x2),self.conv3b(x2)],1)
        
        x3 = self.dropout(self.softmax(x1*x2))
        
        return self.conv4(x0+x1+x2+x3)
    

class FRCM(nn.Module):

    def __init__(self,ch_ins,ch_out,n_sides=11):
        super(FRCM,self).__init__()

        self.reducers = nn.ModuleList([
            nn.Conv2d(ch_in, ch_out, kernel_size=1)
            for ch_in in ch_ins
        ])

        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        self.gn = nn.GroupNorm(1, ch_out)
        
        self.fused = nn.Conv2d(ch_out*n_sides, ch_out, kernel_size=1)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        
        for m in self.reducers:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fused.weight, std=0.01)
        nn.init.constant_(self.fused.bias, 0)

    def get_weight(self):
        return [self.fused.weight]
    
    def get_bias(self):
        return [self.fused.bias]

    def forward_sides(self, sides, img_shape):
        # pass through base_model and store intermediate activations (sides)
        late_sides = []
        for x, conv in zip(sides, self.reducers):
            x = F.interpolate(conv(x), size=img_shape, mode='bilinear', align_corners=True)
            x = self.gn(self.prelu(x))
            late_sides.append(x)

        return late_sides

    def forward(self, img_shape, sides):
        late_sides = self.forward_sides(sides, img_shape)
        
        late_sides1 = torch.cat([late_sides[0],late_sides[1],late_sides[2],late_sides[3],late_sides[4],
                                 late_sides[5],late_sides[6],late_sides[7],late_sides[8],late_sides[9],
                                 late_sides[10]],1)

        fused = self.prelu(self.fused(late_sides1))
        late_sides.append(fused)

        return late_sides


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


class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // reduction, ch_out, kernel_size=1, bias=False),
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
            
        self.sem1 = SEM(ch_out, reduction=reduction)
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


class PFM(nn.Module):
    def __init__(self, ch_in, ch_out, ch_out_3x3e, pool_ch_out, EEM_ch_out, reduction, shortcut=False):
        super(PFM, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.03),
            nn.GroupNorm(4, ch_out)
            )
        ch_in1 = ch_out
        
        # 3x3 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out, kernel_size=3,padding=1,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, ch_out)
            )
        # 3x3 conv extended branch
        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out_3x3e, kernel_size=3,padding=1,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out_3x3e, init=0.),
            nn.GroupNorm(4, ch_out_3x3e)
            )
        # 3x3 pool branch
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in1, pool_ch_out, kernel_size=1,padding=0,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, pool_ch_out)
            )
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out+ch_out_3x3e, kernel_size=1,padding=0,groups=4,bias=False),
                nn.GroupNorm(4, ch_out+ch_out_3x3e)
                )
        
        self.EEM = EEM(ch_in1,EEM_ch_out,kernel=3,groups=ch_in1,reduction=reduction[0])
        self.sem1 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.sem2 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.prelu = nn.PReLU(num_parameters=ch_out+ch_out_3x3e, init=0.03)

    def forward(self, x, shortcut=False): 
        x1 = self.reducer(x)
        
        b1 = self.b1(x1) 
        b2 = self.b2(x1+b1)
        b3 = self.b3(x1)
        eem = self.EEM(x1)
        
        y1 = torch.cat([x1+b1+b3+eem,b2], 1)
        y2 = self.sem1(y1)
        
        if shortcut:
            x = self.shortcut(x)
        y3 = self.sem2(x)
        
        return self.prelu(x+y1+y2+y3)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiScaleAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # Flatten spatial dimensions
        x, _ = self.attention(x, x, x)  # Apply self-attention
        x = x.permute(0, 2, 1).view(b, c, h, w)  # Reshape back
        return x

@HEADS.register_module()  
class MultiScaleFPN(BaseDecodeHead):
    def __init__(self, 
            encoder_channels,
            n_blocks = 4,
            upsampling=1,
            size_shape=(512, 512),
            out_decode=256,
            **kwargs):
        super(MultiScaleFPN, self).__init__(**kwargs)
        # # Backbone: ResNet-50
        # resnet = resnet50(pretrained=True)
        # self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Conv1 + Layer1
        # self.layer2 = resnet.layer2
        # self.layer3 = resnet.layer3
        # self.layer4 = resnet.layer4
        
        self.fpn_blocks = nn.ModuleList(
            [nn.Conv2d(encoder_channels[i], out_decode, kernel_size=1) for i in range(n_blocks)]
        )
        # # Feature Pyramid Network
        # self.fpn_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        # self.fpn_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        # self.fpn_conv3 = nn.Conv2d(1024, 256, kernel_size=1)
        # self.fpn_conv4 = nn.Conv2d(2048, 256, kernel_size=1)
        
        # Multi-Scale Attention
        self.attention1 = MultiScaleAttention(out_decode, num_heads=4)
        self.attention2 = MultiScaleAttention(out_decode, num_heads=4)
        self.attention3 = MultiScaleAttention(out_decode, num_heads=4)
        self.attention4 = MultiScaleAttention(out_decode, num_heads=4)
        
        # Final Segmentation Head
        self.segmentation_head = nn.Conv2d(out_decode, self.out_channels, kernel_size=1)
    
    def forward(self, *features):
        # # Backbone
        # c1 = self.layer1(x)  # Output: (B, 256, H/4, W/4)
        # c2 = self.layer2(c1)  # Output: (B, 512, H/8, W/8)
        # c3 = self.layer3(c2)  # Output: (B, 1024, H/16, W/16)
        # c4 = self.layer4(c3)  # Output: (B, 2048, H/32, W/32)
        
        features = features[0][::-1]

        p = []
        for i, decoder_block in enumerate(reversed(self.fpn_blocks)):
            if len(p) == 0:
                p.append(decoder_block(features[i])) 
            else:
                p.append(decoder_block(features[i]) + F.interpolate(p[-1], scale_factor=2, mode="nearest")) 

        # # FPN
        # p4 = self.fpn_conv4(c4)  # Reduce channels
        # p3 = self.fpn_conv3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        # p2 = self.fpn_conv2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        # p1 = self.fpn_conv1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")
        
        # Multi-Scale Attention
        p1 = self.attention1(p[3])
        p2 = self.attention2(p[2])
        p3 = self.attention3(p[1])
        p4 = self.attention4(p[0])
        
        # Upsample and Aggregate
        out = F.interpolate(p4, scale_factor=8, mode="bilinear", align_corners=False)
        out += F.interpolate(p3, scale_factor=4, mode="bilinear", align_corners=False)
        out += F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        out += p1
        
        # Segmentation Head
        out = self.segmentation_head(out)
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)  # Match input size
        return out

# Test the Model
if __name__ == "__main__":
    model = MultiScaleFPN(num_classes=1)  # Example: 21 classes for PASCAL VOC
    x = torch.randn(2, 3, 256, 256)  # Input: Batch of 2, 3-channel 256x256 images
    out = model(x)
    print("Output shape:", out.shape)  # Should be (2, 21, 256, 256)

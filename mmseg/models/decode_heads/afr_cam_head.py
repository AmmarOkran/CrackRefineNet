import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class StructureTextureFusion(nn.Module):
    def __init__(self, in_channels):
        super(StructureTextureFusion, self).__init__()
        
        # Define Sobel filters as conv layers with depthwise setting
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)

        # Repeat kernels for all channels
        sobel_kernel_x = sobel_kernel_x.repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(in_channels, 1, 1, 1)

        self.sobel_x.weight.data = sobel_kernel_x
        self.sobel_y.weight.data = sobel_kernel_y
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        # Texture pooling branch
        self.texture = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        edge = torch.abs(self.sobel_x(x)) + torch.abs(self.sobel_y(x))  # Shape: [B, C, H, W]
        texture = self.texture(x)  # Local average texture
        fused = torch.cat([edge, texture], dim=1)
        return self.fusion(fused)

    

class CAREEModule(nn.Module):
    def __init__(self, in_ch):
        super(CAREEModule, self).__init__()
        
        # Two Roberts operators
        roberts_1 = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
        roberts_2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
        roberts_bank = torch.stack([roberts_1, roberts_2], dim=0)  # [2, 2, 2]
        
        # Expand to match channels (repeat across in_ch)
        roberts_bank = roberts_bank.unsqueeze(1)  # [2, 1, 2, 2]
        roberts_bank = roberts_bank.repeat(in_ch, 1, 1, 1)  # [2*in_ch, 1, 2, 2]
        
        self.register_buffer('kernel', roberts_bank)
        self.in_ch = in_ch

        # Channel attention block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 16, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # Apply 2 Roberts filters → output shape: [B, 2*C, H-1, W-1]
        edge = F.conv2d(x, weight=self.kernel, groups=self.in_ch)

        # Reshape and aggregate
        edge = edge.view(B, 2, C, H - 1, W - 1).sum(dim=1)  # [B, C, H-1, W-1]

        # Pad to match input size
        edge = F.pad(edge, (0, 1, 0, 1))  # restore to HxW

        ca = self.se(edge)
        return x + ca * edge




class AdaptiveFeatureRefinement(nn.Module):
    def __init__(
        self,
        in_channels,
        use_depthwise=True,
        use_channel_attention=True,
        use_residual=False,
        use_dropout=True,
        use_norm=False,
        dropout_p=0.3,
        norm_groups=4
    ):
        super(AdaptiveFeatureRefinement, self).__init__()

        self.use_channel_attention = use_channel_attention
        self.use_residual = use_residual
        self.use_dropout = use_dropout
        self.use_norm = use_norm

        # Multi-Scale Spatial Refinement (with optional depthwise)
        conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                            groups=in_channels if use_depthwise else 1)
        conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2,
                            groups=in_channels if use_depthwise else 1)
        self.conv3x3 = nn.Sequential(conv3x3, nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(inplace=True))
        self.conv5x5 = nn.Sequential(conv5x5, nn.Conv2d(in_channels, in_channels, 1), nn.ReLU(inplace=True))

        # Channel Attention (SE block)
        if use_channel_attention:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
            self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        # Adaptive gating
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid_gate = nn.Sigmoid()

        # Normalization (optional GroupNorm)
        if self.use_norm:
            self.norm = nn.GroupNorm(norm_groups, in_channels)

        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        # Multi-scale spatial context
        spatial = self.conv3x3(x) + self.conv5x5(x)

        if self.use_norm:
            spatial = self.norm(spatial)

        # Channel Attention
        if self.use_channel_attention:
            channel = self.global_avg_pool(x)
            channel = self.relu(self.fc1(channel))
            channel = self.sigmoid(self.fc2(channel))
        else:
            channel = 1

        # Gating
        gate = self.sigmoid_gate(self.gate(spatial))

        # Final fusion
        out = x * gate * channel

        if self.use_dropout:
            out = self.dropout(out)

        if self.use_residual:
            out = out + x

        return out
    

class ContextAwareModule(nn.Module):
    def __init__(self, in_channels, normalize=False):

        super(ContextAwareModule, self).__init__()
        self.normalize = normalize  # Toggle for normalization
        # Multi-Scale Spatial Attention  ImprovedContextAwareModule
        self.conv1 = nn.Conv2d(2, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)  # Larger kernel
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)  # Even larger

        # Channel Attention (Squeeze-and-Excite)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Compute Spatial Complexity Maps
        complexity_map1 = torch.std(x, dim=1, keepdim=True)  # Variance as complexity
        complexity_map2 = torch.mean(x, dim=1, keepdim=True)  # Mean as complexity
        complexity_map = torch.cat([complexity_map1, complexity_map2], dim=1)

        if self.normalize:
            mean = complexity_map.mean(dim=(2, 3), keepdim=True)
            std = complexity_map.std(dim=(2, 3), keepdim=True) + 1e-6
            complexity_map = (complexity_map - mean) / std

        # Spatial Attention (Multi-Scale Processing)
        spatial_att = self.conv1(complexity_map)
        spatial_att = self.relu(self.conv2(spatial_att))
        spatial_att = self.sigmoid(self.conv3(spatial_att) + self.conv4(spatial_att))  # Multi-scale fusion

        # Channel Attention (SE Block)
        channel_att = self.global_avg_pool(x)
        channel_att = self.relu(self.fc1(channel_att))
        channel_att = self.sigmoid(self.fc2(channel_att))

        # Apply Spatial & Channel Attention
        return x * spatial_att * channel_att# + x  # Residual connection
    
    
class SelectiveFeatureAggregation(nn.Module):
    def __init__(self, in_channels, num_scales):
        super(SelectiveFeatureAggregation, self).__init__()
        # self.weights = nn.Parameter(torch.tensor([0.6, 0.5, 0.3, 0.2]))  # Learnable weights for each scale
        self.weights = nn.Parameter(torch.ones(num_scales))  # Learnable weights for each scale
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Optional final adjustment

    def forward(self, features):
        # Resize all feature maps to the size of the first feature map (highest resolution)
        target_size = features[0].shape[2:]  # Height and width of p1
        resized_features = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
        
        # Apply weights and sum the features
        weighted_features = [self.weights[i] * resized_features[i] for i in range(len(resized_features))]
        aggregated_features = torch.sum(torch.stack(weighted_features, dim=0), dim=0)
        
        return self.conv(aggregated_features)


@HEADS.register_module()      
class AFRCAMLASTSegmentationModel(BaseDecodeHead):
    def __init__(
            self, 
            encoder_channels,
            n_blocks = 4,
            upsampling=1,
            size_shape=(512, 512),
            out_decode=256,
            **kwargs):
        super(AFRCAMLASTSegmentationModel, self).__init__(**kwargs)
        self.size_shape = size_shape
        self.upsampling = upsampling
        # FPN
        self.fpn_blocks = nn.ModuleList(
            [nn.Conv2d(encoder_channels[i], out_decode, kernel_size=1) for i in range(n_blocks)]
        )

        # Context-Aware Module
        self.cam1 = ContextAwareModule(out_decode, normalize=True)
        self.cam2 = ContextAwareModule(out_decode, normalize=True)
       

        # Adaptive Feature Refinement
        self.afr1 = AdaptiveFeatureRefinement(256, use_norm=True, use_residual=False)
        self.afr2 = AdaptiveFeatureRefinement(256, use_norm=True, use_residual=False)

        # Simple Decoder (No CAM, AFR, or SFA)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(out_decode, out_decode, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(out_decode, out_decode, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(out_decode, out_decode, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(out_decode, out_decode, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
            nn.ConvTranspose2d(out_decode, out_decode, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_decode),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
        )

        # Updated Selective Feature Aggregation
        self.sfa = SelectiveFeatureAggregation(out_decode, num_scales=4)
        self.sfa_dropout = nn.Dropout2d(p=0.3)

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(out_decode*2, self.out_channels, kernel_size=1)
        
    def forward(self, *features):
        # Backbone
        features = features[0]
    
        # FPN feature projection
        p = [block(f) for block, f in zip(self.fpn_blocks, features)]
    
        p1 = self.afr1(p[3]) #+ p[3]
        # dec1 = F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False)
        dec1 = self.decoder1(p1)

        p2 = self.afr2(p[2] + dec1) #+ p[2]
        # dec2 = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        dec2 = self.decoder2(p2)

        p3 = self.cam1(p[1] + dec2) #+ p[1]
        dec3 = self.decoder3(p3)
        # dec3 = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)

        p4 = self.cam2(p[0] + dec3) #+ p[0]
        dec_out = self.decoder4(p4)

        # Selective Feature Aggregation
        aggregated_features = self.sfa([p4, p3, p2, p1])
        aggregated_features = self.sfa_dropout(aggregated_features)  # Prevent overfitting
        aggregated_features = F.interpolate(aggregated_features, scale_factor=self.upsampling, mode="bilinear", align_corners=False)
        aggregated_features = torch.cat([dec_out, aggregated_features], dim=1)  # Residual connection
        # aggregated_features = dec_out + aggregated_features # Residual connection

        # Segmentation Output
        out = self.segmentation_head(aggregated_features)
        return out


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 512, 512)  # Batch size 1, RGB input
    model = AFRCAMLASTSegmentationModel(4)  # Binary segmentation
    output = model(input_tensor)
    print(output.shape)  # Expect: torch.Size([1, 1, 256, 256])

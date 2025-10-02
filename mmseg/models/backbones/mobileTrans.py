import torch
import torch.nn as nn

from .module import InvertedResidual, MobileVitBlock

from mmseg.registry import MODELS
from mmengine.model import BaseModule


model_cfg = {
    "xxs":{
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        "layers": [2, 4, 3]
    },
    "xs":{
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "n":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
}

@MODELS.register_module()
class MobileViT(BaseModule):
    def __init__(self, features_list=model_cfg["s"]["features"], 
                d_list=model_cfg["s"]["d"], transformer_depth=model_cfg["s"]["layers"], 
                expansion=model_cfg["s"]["expansion_ratio"], num_classes = 1000):
        super(MobileViT, self).__init__()
        # self.encoder = encoder

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = features_list[0], kernel_size = 3, stride = 2, padding = 1),
            InvertedResidual(in_channels = features_list[0], out_channels = features_list[1], stride = 1, expand_ratio = expansion),
        )
    

        self.stage1 = nn.Sequential(
            InvertedResidual(in_channels = features_list[1], out_channels = features_list[2], stride = 2, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[2], stride = 1, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[3], stride = 1, expand_ratio = expansion)
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels = features_list[3], out_channels = features_list[4], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[4], out_channels = features_list[5], d_model = d_list[0],
                           layers = transformer_depth[0], mlp_dim = d_list[0] * 2)
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels = features_list[5], out_channels = features_list[6], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[6], out_channels = features_list[7], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 4)
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels = features_list[7], out_channels = features_list[8], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[8], out_channels = features_list[9], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 4),
            nn.Conv2d(in_channels = features_list[9], out_channels = features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )


        self.dconv1 = nn.ConvTranspose2d(640, 192, 4, 2, 1)  #1920
 
        self.dconv2 = nn.ConvTranspose2d(320, 128, 4, 2, 1)
  
        self.dconv3 = nn.ConvTranspose2d(224, 96, 4, 2, 1)

        self.dconv4 = nn.ConvTranspose2d(160, 48, 4, 2, 1)
       
        self.dconv5 = nn.ConvTranspose2d(80, num_classes, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(192)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.batch_norm4 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(True)

        self.rs = nn.Conv2d(1280, 3, kernel_size=3,padding=1, bias=False)

    def forward(self, x):                             
        # print('x', x.shape)
        # Stem
        x1 = self.stem(x)                               #[4, 16, 128, 128]
        x1 = nn.functional.interpolate(x1, size=(128,128), mode='bilinear', align_corners=True)
        # Body
        x2 = self.stage1(x1)                             #[4, 24, 64, 64]  
        x3 = self.stage2(x2)                             #[4, 48, 32, 32]
        x4 = self.stage3(x3)                             #[4, 64, 16, 16]
        x4 = nn.functional.interpolate(x4, size=(16,16), mode='bilinear', align_corners=True)
        x5 = self.stage4(x4)                             #[4, 320, 8, 8]

        # Decoder
        d1_ = self.batch_norm(self.dconv1(self.relu(x5)))                     # 4, 384,  16, 16
        d1 = torch.cat((d1_, x4), 1) 

        d2_ = self.batch_norm1(self.dconv2(self.relu(d1)))                    # 2, 384, 28, 28
        d2_ = nn.functional.interpolate(d2_, size=(32,32), mode='bilinear', align_corners=True)
        d2 = torch.cat((d2_, x3), 1) 

        d3_ = self.batch_norm2(self.dconv3(self.relu(d2)))                    # 2, 192, 56, 56
        d3_ = nn.functional.interpolate(d3_, size=(64,64), mode='bilinear', align_corners=True)
        d3 = torch.cat((d3_, x2), 1)                                          # 2, 288, 56, 56
        # print('decoder_3', d3.shape)
                                                  

        d4_ = self.batch_norm3(self.dconv4(self.relu(d3)))                    # 2, 96, 112, 112
        d4_ = nn.functional.interpolate(d4_, size=(128,128), mode='bilinear', align_corners=True)
        d4 = torch.cat((d4_, x1), 1)
        # print('decoder_4', d4.shape)

        d5 = self.dconv5(self.relu(d4))
        
        return [d5]


# def MobileViT_XXS(img_size = 256, num_classes = 1):
#     cfg_xxs = model_cfg["xxs"]
#     model_xxs = MobileViT(img_size, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"], num_classes)
#     return model_xxs

# def MobileViT_XS(img_size = 256, num_classes = 1000):
#     cfg_xs = model_cfg["xs"]
#     model_xs = MobileViT(img_size, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"], num_classes)
#     return model_xs

# def MobileViT_S(img_size = 256, num_classes = 32):
#     cfg_s = model_cfg["s"]
#     model_s = MobileViT(img_size, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"], num_classes)
#     return model_s


# if __name__ == "__main__":
#     model = MobileViT_S().cuda()
#     img = torch.randint(0, 2, (1, 3, 256, 256)).type(torch.float32)
#     pred = model(img.cuda())
#     print(pred)
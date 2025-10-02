import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from mmseg.registry import MODELS
from .utils import weighted_loss

# @weighted_loss
# def l1_loss(pred, target):
#     # assert pred.size() == target.size() and target.numel() > 0
#     # loss = torch.mean(torch.abs(pred - target.unsqueeze(1)))
#     loss = F.l1_loss(pred, target.unsqueeze(1))
#     return loss

@MODELS.register_module()
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', 
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_mse'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index
        self.criterion = nn.MSELoss()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # pred = F.softmax(pred, dim=1)
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
            # transforms.ToPILImage()
        ])
        
        # target = denormalize(target)
        loss = self.loss_weight * self.criterion(pred, target)
        # loss1 = self.criterion1(pred, target)
        return loss# + loss1
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

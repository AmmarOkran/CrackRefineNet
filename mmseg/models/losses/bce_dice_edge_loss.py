
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.registry import MODELS


def edge_consistency_loss(pred, target):
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if pred.is_cuda:
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()

    # Compute edge gradients for prediction and target
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target.float().unsqueeze(1), sobel_x, padding=1)
    target_grad_y = F.conv2d(target.float().unsqueeze(1), sobel_y, padding=1)

    # Edge consistency loss (MSE between gradients)
    edge_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    edge_loss_y = F.mse_loss(pred_grad_y, target_grad_y)

    return edge_loss_x + edge_loss_y


@MODELS.register_module()
class CombinedLoss(nn.Module):
    def __init__(
            self,
            pos_weight=5, 
            edge_weight=1.0,
            loss_weight=1.0,
            loss_name='loss_combined'):
        super(CombinedLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.pos_weight = pos_weight
        self.edge_weight = edge_weight

    def forward(self, outputs, targets, **kwargs):
        # Weighted Binary Cross-Entropy Loss
        
        weights = targets * (self.pos_weight - 1) + 1
        bce_loss = F.binary_cross_entropy_with_logits(outputs.squeeze(1), targets.float(), weight=weights)

        # Dice Loss
        smooth = 1e-5
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

        # Edge-Consistency Loss
        edge_loss = edge_consistency_loss(outputs, targets)

        # Combine losses
        total_loss = bce_loss + dice_loss + self.edge_weight * edge_loss
        return total_loss


    @property
    def loss_name(self):
        return self._loss_name
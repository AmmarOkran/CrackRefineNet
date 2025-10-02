
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@LOSSES.register_module()
class MSEDiceLoss(nn.Module):
    def __init__(
            self, 
            weight_mse=0.5, 
            weight_dice=0.5, 
            smooth=1,
            loss_weight=1.0,
            loss_name='loss_mse_dice',
            **kwargs):
        """
        Combines Mean Squared Error (MSE) and soft Dice loss.
        
        Args:
            weight_mse (float): Weight for the MSE component.
            weight_dice (float): Weight for the Dice component.
            smooth (float): Smoothing constant to prevent division by zero.
        """
        super(MSEDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.weight_mse = weight_mse
        self.weight_dice = weight_dice
        self.smooth = smooth

    def dice_loss(self, preds, targets):
        """
        Computes soft Dice loss between continuous predictions and targets.
        Both preds and targets are assumed to be in [0, 1].
        """
        # Flatten predictions and targets per sample
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

    def forward(self, preds, targets):
        # Assuming preds are already in [0, 1] because sigmoid is applied upstream
        mse_loss = F.mse_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        combined_loss = self.weight_mse * mse_loss + self.weight_dice * dice
        return combined_loss
    

    @property
    def loss_name(self):
        return self._loss_name
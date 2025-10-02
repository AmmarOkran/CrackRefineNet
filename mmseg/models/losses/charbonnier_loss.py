
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """
    A smooth L1 loss function to prevent large penalties on small errors.
    """
    def __init__(
            self, 
            epsilon=1e-6,
            weight_bce=None,
            loss_weight=1.0,
            loss_name='loss_Charb',
            **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.weight_bce=weight_bce
        self.epsilon = epsilon  # Smooth term

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))  # Smooth L1
        return loss

    @property
    def loss_name(self):
        return self._loss_name
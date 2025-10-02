
import torch
import torch.nn as nn

from mmseg.registry import MODELS



@MODELS.register_module()
class DiceBinaryLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_name='loss_dicebinary'):
        super(DiceBinaryLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, outputs, targets, **kwargs):
        """
        Args:
            outputs: tensor of shape (B, 1, H, W) or (B, H, W)
            targets: tensor of shape (B, 1, H, W) or (B, H, W)
        Returns:
            Dice loss averaged over batch.
        """
        smooth = 1e-5
        outputs = torch.sigmoid(outputs)
        
        # Flatten per image
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (outputs * targets).sum(dim=1)
        union = outputs.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + smooth) / (union + smooth)
        loss = 1 - dice  # Dice loss

        return self.loss_weight * loss.mean()


    @property
    def loss_name(self):
        return self._loss_name
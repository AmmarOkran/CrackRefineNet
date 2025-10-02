
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


def edge_filter(x, filter_type='sobel'):
    """
    Apply edge detection filters: Sobel, Scharr, Laplacian, or Hybrid.
    """
    if filter_type == 'sobel':
        return sobel_filter(x)
    elif filter_type == 'scharr':
        return scharr_filter(x)
    elif filter_type == 'laplacian':
        return laplacian_filter(x)
    elif filter_type == 'hybrid':
        return hybrid_filter(x)
    else:
        raise ValueError("Unsupported filter_type. Choose 'sobel', 'scharr', 'laplacian', or 'hybrid'.")

def sobel_filter(x, epsilon=1e-6):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(x, sobel_x, padding=1)
    edge_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(edge_x ** 2 + edge_y ** 2 + epsilon)

def scharr_filter(x, epsilon=1e-6):
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                            dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    edge_x = F.conv2d(x, scharr_x, padding=1)
    edge_y = F.conv2d(x, scharr_y, padding=1)
    return torch.sqrt(edge_x ** 2 + edge_y ** 2 + epsilon)

def laplacian_filter(x):
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    edge = F.conv2d(x, laplacian, padding=1)
    return torch.abs(edge)


def hybrid_filter(x):
    """
    Hybrid filter: Combines Sobel and Laplacian edge maps.
    """
    sobel_edges = sobel_filter(x)
    laplacian_edges = laplacian_filter(x)
    hybrid_edges = (sobel_edges + laplacian_edges) / 2.0  # Average the two edge maps
    return hybrid_edges


@weighted_loss
def edge(prediction, target, filter_type='sobel', epsilon=1e-6):
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Ensure target is float
        target = target.float()
        
        # Convert logits to probabilities
        prediction = torch.sigmoid(prediction)

        # Compute edge maps
        pred_edges = edge_filter(prediction, filter_type=filter_type)
        target_edges = edge_filter(target, filter_type=filter_type)

        # L1 loss between edge maps
        edge_loss = F.l1_loss(pred_edges, target_edges)
        return edge_loss 


@LOSSES.register_module()
class EdgeLoss(nn.Module):
    
    def __init__(self,
                 filter_type='sobel',
                 loss_weight=1.0,
                 loss_name='loss_edge',
                 epsilon=1e-6,
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.filter_type = filter_type
        self.epsilon = epsilon

    def forward(self, pred, target, **kwargs):
        # Combine losses
        loss = self.loss_weight * edge(pred, target, filter_type=self.filter_type, 
                                       epsilon=self.epsilon)
        return loss

    @property
    def loss_name(self):
        return self._loss_name

import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    """
    Applies the Generalized Mean (GeM) pooling operation to the input tensor x.

    Args:
        x (torch.Tensor): The input tensor.
        p (float, optional): The exponent value for the GeM pooling. Defaults to 3.
        eps (float, optional): Eps to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The result of the GeM pooling operation.
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) module.

    Attributes:
        p (float or torch.Tensor): The power parameter for GeM pooling.
        eps (float): Small epsilon value for numerical stability.

    Methods:
        __init__(self, p=3, eps=1e-6, p_trainable=False): Constructor
        forward(self, x): Forward pass of the GeM module.
    """

    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        """
        Constructor.

        Args:
            p (float or torch.Tensor): The power parameter for GeM pooling. Defaults to 3.
            eps (float): Small epsilon value for numerical stability. Defaults to 1e-6.
            p_trainable (bool): Whether the power parameter is trainable. Defaults to False.
        """
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the GeM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after GeM pooling.
        """
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

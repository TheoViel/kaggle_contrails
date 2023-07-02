import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss

from training.lovasz import lovasz_loss


class ContrailLoss(nn.Module):
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        aux_loss_weight (float): Weight for the auxiliary loss.
        ousm_k (int): Number of samples to exclude in the OUSM variant. Defaults to 0.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.

    Methods:
        __init__(self, config, device="cuda"): Constructor.
        prepare(self, pred, y): Prepares the predictions and targets for loss computation.
        forward(self, pred, pred_aux, y, y_aux): Computes the loss.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.ousm_k = config.get("ousm_k", 0)
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "lovasz":
            self.loss = lovasz_loss
        elif config["name"] == "dice":
            loss = DiceLoss(mode="binary", smooth=1)
        else:
            raise NotImplementedError

        self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Predictions.
            y (torch.Tensor): Targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze()
            y_aux = y_aux.squeeze()
        else:  # bce, lovasz
            y = y.float()
            pred = pred.float().view(y.size())
            
        y_aux = y_aux.float()
        pred_aux = pred_aux.float().view(y_aux.size())

        if self.eps and self.config["name"] == "bce":
            y = torch.clamp(y, self.eps, 1 - self.eps)

        return pred, pred_aux, y, y_aux

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        loss = self.loss(pred, y)

        if self.ousm_k:
            raise NotImplementedError
            _, idxs = loss.topk(y.size(0) - self.ousm_k, largest=False)
            loss = loss.index_select(0, idxs)

        if len(loss.size()) >= 1:
            loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss
        
        loss_aux = self.loss_aux(pred_aux, y_aux).mean()
        return (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux

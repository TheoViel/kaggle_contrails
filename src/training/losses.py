import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss

from training.lovasz import lovasz_sym


class FocalLoss(nn.Module):
    """
    Focal Loss implementation.
    Methods:
        __init__(self, gamma=2, reduction="mean"):
            Constructor for the FocalLoss class.
        forward(self, input, target):
            Forward pass of the focal loss.

    Attributes:
        gamma (float): The focal loss exponent.
        reduction (str): The reduction applied to the output.
    """
    def __init__(self, gamma=2, reduction="mean"):
        """
        Constructor.

        Args:
            gamma (float, optional): The focal loss exponent. Defaults to 2.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute the focal loss.

        Args:
            input (torch.Tensor): The predicted logits or probabilities.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed focal loss.
        """
        # n = input.shape[-1]
        input = input.view(-1).float()
        target = target.view(-1).float()
        loss = -target * F.logsigmoid(input) * torch.exp(
            self.gamma * F.logsigmoid(-input)
        ) - (1.0 - target) * F.logsigmoid(-input) * torch.exp(
            self.gamma * F.logsigmoid(input)
        )
        loss = 100 * loss
        return loss.mean() if self.reduction == "mean" else loss


class LovaszFocalLoss(nn.Module):
    """
    Combined Lovasz and Focal Loss implementation.

    Methods:
        __init__(self, gamma=2, alpha=0.7, reduction="mean"):
            Constructor for the LovaszFocalLoss class.
        forward(self, x, y):
            Compute the combined Lovasz and Focal loss.

    Attributes:
        focal_loss (FocalLoss): The focal loss component.
        alpha (float): The weight factor for the Lovasz loss.
    """
    def __init__(self, gamma=2, alpha=0.7, reduction="mean"):
        """
        Constructor.

        Args:
            gamma (float, optional): The focal loss exponent. Defaults to 2.
            alpha (float, optional): Weight factor for the Lovasz loss. Defaults to 0.7.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.focal_loss = FocalLoss(gamma, reduction=reduction)
        self.alpha = alpha

    def forward(self, x, y):
        """
        Compute the combined Lovasz and Focal loss.

        Args:
            x (torch.Tensor): The predicted logits or probabilities.
            y (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed combined loss.
        """
        return self.focal_loss(x, y) + self.alpha * lovasz_sym(x, y, per_image=False)


class LovaszBCELoss(nn.Module):
    """
    Lovasz BCE Loss for semantic segmentation tasks.

    This loss combines the binary cross-entropy (BCE) loss and the Lovasz loss.
    The Lovasz loss is used for tasks with imbalanced data, which helps prevent
    overfitting in situations where certain classes are rare.

    Args:
        alpha (float, optional): Weighting parameter for the Lovasz loss.
            Defaults to 0.01.
        reduction (str, optional): Specifies the reduction to apply to the loss.
            Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.
    """
    def __init__(self, alpha=0.01, reduction="mean"):
        """
        Constructor.

        Args:
            alpha (float, optional): Weighting parameter for the Lovasz loss.
                Defaults to 0.01.
            reduction (str, optional): Specifies the reduction to apply to the loss.
                Can be 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.alpha = alpha

    def forward(self, x, y):
        """
        Compute the Lovasz BCE loss.

        Args:
            x (torch tensor): Predicted logits.
            y (torch tensor): Ground truth labels.

        Returns:
            torch tensor: Computed Lovasz BCE loss.
        """
        return self.bce(x, y) + self.alpha * lovasz_sym(x, y, per_image=False)


class ContrailLoss(nn.Module):
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        shape_loss_w (float): Weight for the shape loss.
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.
        shape_loss (nn.Modulee): Shape loss function.

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

        self.shape_loss_w = config['shape_loss_w']
        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "lovasz":
            self.loss = lovasz_sym
        elif config["name"] == "focal":
            self.loss = FocalLoss(gamma=2, reduction="mean")
        elif config["name"] == "lovasz_focal":
            self.loss = LovaszFocalLoss(reduction="mean")
        elif config["name"] == "lovasz_bce":
            self.loss = LovaszBCELoss(reduction="mean")
        elif config["name"] == "dice":
            self.loss = DiceLoss(mode="binary", smooth=1)
        else:
            raise NotImplementedError

        self.loss_aux = nn.BCEWithLogitsLoss(reduction="mean")

        if config['shape_loss'] == "bce":
            self.shape_loss = nn.BCEWithLogitsLoss(reduction="mean")
        else:  # mse
            self.shape_loss = nn.MSELoss(reduction="mean")

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze()
            y_aux = y_aux.squeeze()
        else:  # bce, lovasz, focal
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
        Main predictions are masks for the segmentation task.
        They are of size [BS x C x H x W] where C=7 if the shape loss is used else 1
        Auxiliary predictions are for the (optional) classification task.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        y_shape = None
        if self.shape_loss_w and y.size(1) > 6:
            pred_shape, y_shape = pred[:, -6:].contiguous(), y[:, -6:].contiguous()
            pred, y = pred[:, :-6].contiguous(), y[:, :-6].contiguous()
        else:
            pred = pred[:, :y.size(1)]

        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        loss = self.loss(pred, y)

        if self.shape_loss_w and y_shape is not None:
            shape_loss = self.shape_loss(pred_shape, y_shape)
            loss += self.shape_loss_w * shape_loss

        if len(loss.size()) >= 1:
            loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss

        loss_aux = self.loss_aux(pred_aux, y_aux)
        return (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux

import numpy as np


def dice_score(preds, truths, eps=1e-8):
    """
    Dice metric.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    preds = preds.reshape(-1) > 0
    truths = truths.reshape(-1) > 0
    intersect = (preds & truths).sum(-1)
    union = preds.sum(-1) + truths.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return dice


def dice_score_torch(preds, truths, eps=1e-8):
    """
    Dice metric in torch.

    Args:
        pred (torch tensor): Predictions.
        truth (torch tensor): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    preds = preds.view(-1) > 0
    truths = truths.contiguous().view(-1) > 0
    intersect = (preds & truths).sum(-1)
    union = preds.sum(-1) + truths.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return float(dice)

import torch
import numpy as np
from util.metrics import dice_score_torch


class SegmentationMeter1:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, threshold=0.5):
        """
        Constructor

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """
        self.threshold = threshold
        self.reset()

    def update(self, y, y_aux, y_pred, y_pred_aux):
        """
        Updates the meter.
        """
        bs = y.size(0)
        y_pred = y_pred.view(bs, -1) > self.threshold
        
        y = y.contiguous().view(bs, -1) > 0

        self.unions.append(y_pred.sum(-1) + y.sum(-1))
        self.intersections.append((y_pred & y).sum(-1))
        self.accs.append(
            ((y_pred_aux.view(-1) > self.threshold) == (y_aux.view(-1) > 0)).float()
        )

    def reset(self):
        """
        Resets everything.
        """
        self.unions = []
        self.intersections = []
        self.accs = []

        
class SegmentationMeter:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, threshold=0.25):
        """
        Constructor

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """
        self.threshold = threshold
#         self.thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.thresholds = np.round(np.arange(0.2, 0.6, 0.01), 2)
        self.reset()

    def update(self, y, y_aux, y_pred, y_pred_aux):
        """
        Updates the meter.
        """
#         bs = y.size(0)
        y_pred = y_pred.view(1, -1) 
        y = y.contiguous().view(1, -1) > 0

        for th in self.thresholds:
            self.unions[th] += ((y_pred > th).sum(-1) + y.sum(-1)).int()
            self.intersections[th] += (((y_pred > th) & y).sum(-1)).int()

        self.accs.append(
            ((y_pred_aux.view(-1) > self.threshold) == (y_aux.view(-1) > 0)).float()
        )

    def reset(self):
        """
        Resets everything.
        """
        self.unions = {th: torch.zeros(1, dtype=torch.int).cuda() for th in self.thresholds}
        self.intersections = {th: torch.zeros(1, dtype=torch.int).cuda() for th in self.thresholds}
        self.accs = []

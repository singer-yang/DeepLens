"""PSNR loss function."""

import numpy as np
import torch
import torch.nn as nn


class PSNRLoss(nn.Module):
    """Peak Signal-to-Noise Ratio (PSNR) loss."""

    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        """Initialize PSNR loss.
        
        Args:
            loss_weight: Weight for the loss.
            reduction: Reduction method, only "mean" is supported.
            toY: Whether to convert RGB to Y channel.
        """
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        """Calculate PSNR loss.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            
        Returns:
            PSNR loss.
        """
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        ) 
"""SSIM loss function."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss."""

    def __init__(self, window_size=11, size_average=True):
        """Initialize SSIM loss.
        
        Args:
            window_size: Size of the window.
            size_average: Whether to average the loss.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def forward(self, pred, target):
        """Calculate SSIM loss.
        
        Args:
            pred: Predicted tensor.
            target: Target tensor.
            
        Returns:
            1 - SSIM value.
        """
        return 1 - self._ssim(pred, target)

    def _gaussian(self, window_size, sigma):
        """Create a Gaussian window.
        
        Args:
            window_size: Size of the window.
            sigma: Standard deviation.
            
        Returns:
            Gaussian window.
        """
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create a window for SSIM calculation.
        
        Args:
            window_size: Size of the window.
            channel: Number of channels.
            
        Returns:
            Window tensor.
        """
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        """Calculate SSIM value.
        
        Args:
            img1: First image.
            img2: Second image.
            
        Returns:
            SSIM value.
        """
        (_, channel, _, _) = img1.size()
        window = self.window
        window = window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1) 
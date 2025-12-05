"""Dead pixel correction (DPC)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeadPixelCorrection(nn.Module):
    """Dead pixel correction (DPC).

    Detects and corrects dead/stuck pixels by comparing each pixel to its
    neighbors and replacing outliers with a local mean value.

    Note: Uses differentiable operations (mean instead of median, soft mask).

    Reference:
        [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/dpc.py
    """

    def __init__(self, threshold=0.1, kernel_size=3, soft_blend=True, temperature=0.01):
        """Initialize dead pixel correction.

        Args:
            threshold: Threshold for detecting dead pixels (as fraction of max value).
            kernel_size: Size of the kernel for correction (must be odd).
            soft_blend: If True, use differentiable soft blending. If False, use hard threshold.
            temperature: Temperature for soft sigmoid blending (lower = sharper transition).
        """
        super().__init__()
        self.threshold = threshold
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.soft_blend = soft_blend
        self.temperature = temperature

        # Pre-compute averaging kernel (excluding center pixel)
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2
        kernel[0, 0, center, center] = 0  # Exclude center pixel
        kernel = kernel / kernel.sum()  # Normalize
        self.register_buffer("avg_kernel", kernel)

    def forward(self, bayer):
        """Dead Pixel Correction (differentiable).

        Args:
            bayer (torch.Tensor): Input bayer image [B, 1, H, W], data range [0, 1].

        Returns:
            bayer_corrected (torch.Tensor): Corrected bayer image [B, 1, H, W].
        """
        padding = self.kernel_size // 2

        # Compute local mean (excluding center pixel) - differentiable
        local_mean = F.conv2d(bayer, self.avg_kernel.to(bayer.dtype), padding=padding)

        # Compute difference from local mean
        diff = torch.abs(bayer - local_mean)

        if self.soft_blend:
            # Soft differentiable blending using sigmoid
            # blend_weight approaches 1 when diff >> threshold (use local_mean)
            # blend_weight approaches 0 when diff << threshold (use original)
            blend_weight = torch.sigmoid((diff - self.threshold) / self.temperature)
            result = (1 - blend_weight) * bayer + blend_weight * local_mean
        else:
            # Hard threshold (not differentiable through the mask)
            mask = (diff > self.threshold).float()
            result = (1 - mask) * bayer + mask * local_mean

        return result

    def reverse(self, bayer):
        """Reverse dead pixel correction (identity).

        Note: Dead pixel correction is a lossy operation that cannot be reversed.
        This returns the input unchanged.

        Args:
            bayer (torch.Tensor): Input bayer image [B, 1, H, W].

        Returns:
            bayer (torch.Tensor): Input unchanged.
        """
        # Dead pixel correction cannot be reversed; return input as-is
        return bayer

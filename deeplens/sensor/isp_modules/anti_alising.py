"""Anti-aliasing filter (AAF)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasingFilter(nn.Module):
    """Anti-Aliasing Filter (AAF). 
    
    Anti-aliasing filter is applied to raw Bayer data to reduce moiré patterns
    and aliasing artifacts before demosaicing.

    Reference:
        [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/aaf.py
    """

    def __init__(self, method="weighted_average", kernel_size=3):
        """Initialize the Anti-Aliasing Filter.

        Args:
            method (str): Filtering method. Options: "weighted_average", "gaussian", "none", or None.
            kernel_size (int): Size of the filter kernel (must be odd).
        """
        super(AntiAliasingFilter, self).__init__()
        self.method = method
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # Pre-compute kernels
        if method == "weighted_average":
            # Weighted average kernel: center pixel gets higher weight
            kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
            center = self.kernel_size // 2
            kernel[0, 0, center, center] = 8.0
            kernel = kernel / kernel.sum()
            self.register_buffer("kernel", kernel)
        elif method == "gaussian":
            # Gaussian kernel
            sigma = self.kernel_size / 6.0
            x = torch.arange(self.kernel_size) - self.kernel_size // 2
            x = x.float()
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d / kernel_2d.sum()
            self.register_buffer(
                "kernel", kernel_2d.view(1, 1, self.kernel_size, self.kernel_size)
            )

    def forward(self, bayer):
        """Apply anti-aliasing filter to remove moiré pattern.

        Args:
            bayer: Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            Filtered bayer tensor of same shape as input.
        """
        if self.method is None or self.method == "none":
            return bayer

        if self.method in ["weighted_average", "gaussian"]:
            padding = self.kernel_size // 2
            # Apply convolution filter
            filtered = F.conv2d(bayer, self.kernel.to(bayer.dtype), padding=padding)
            return filtered

        else:
            raise ValueError(f"Unknown anti-aliasing method: {self.method}")

    def reverse(self, bayer):
        """Reverse anti-aliasing filter (approximation).

        Note: Anti-aliasing is a lossy operation, so perfect reversal is not possible.
        This returns the input unchanged as an approximation.

        Args:
            bayer: Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            Input tensor unchanged.
        """
        # Anti-aliasing filtering is lossy; we cannot perfectly reverse it
        # Return input unchanged as best approximation
        return bayer

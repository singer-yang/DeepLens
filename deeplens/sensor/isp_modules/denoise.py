"""Denoise.

Reference:
    [1] It is important to remove sensor noise before applying demosaic/CFA, otherwise the CFA will produce zipper noise which is harder to remove. "Thus, it is desirable to suppress the zipper noise in the interpolation stage instead of using noise reduction filter after color interpolation", page 31, Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2.
    [2] Denoise can also be implemented with deep learning methods, replacing the classical denoise filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoise(nn.Module):
    """Noise reduction (differentiable).

    Applies denoising filters to reduce sensor noise in the image.
    Supports Gaussian filtering (differentiable) and bilateral filtering.

    Note: Median filtering is NOT differentiable, so we use Gaussian or bilateral instead.
    """

    def __init__(self, method="gaussian", kernel_size=3, sigma=0.5, sigma_color=0.1):
        """Initialize denoise.

        Args:
            method: Noise reduction method: "gaussian", "bilateral", or None.
            kernel_size: Size of the kernel (must be odd).
            sigma: Standard deviation for spatial Gaussian kernel.
            sigma_color: Standard deviation for color/intensity similarity (bilateral only).
        """
        super().__init__()
        self.method = method
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.sigma_color = sigma_color

        # Pre-compute Gaussian kernel
        kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)
        self.register_buffer("gaussian_kernel", kernel)

    def forward(self, img):
        """Apply denoise (differentiable).

        Args:
            img (torch.Tensor): Input tensor of shape [B, C, H, W], data range [0, 1].

        Returns:
            img_filtered (torch.Tensor): Denoised image, data range [0, 1].
        """
        if self.method is None or self.method == "none":
            return img

        if self.method == "gaussian":
            img_filtered = self._gaussian_filter(img)

        elif self.method == "bilateral":
            img_filtered = self._bilateral_filter(img)

        else:
            raise ValueError(f"Unknown noise reduction method: {self.method}")

        return img_filtered

    def _gaussian_filter(self, img):
        """Apply Gaussian filter (differentiable).

        Args:
            img: Input tensor of shape [B, C, H, W].

        Returns:
            Filtered tensor of same shape.
        """
        padding = self.kernel_size // 2
        C = img.shape[1]
        kernel = self.gaussian_kernel.to(img.dtype).expand(
            C, 1, self.kernel_size, self.kernel_size
        )
        img_filtered = F.conv2d(img, kernel, padding=padding, groups=C)
        return img_filtered

    def _bilateral_filter(self, img):
        """Apply bilateral filter (differentiable approximation).

        Bilateral filter preserves edges while smoothing by considering both
        spatial distance and intensity similarity.

        Args:
            img: Input tensor of shape [B, C, H, W].

        Returns:
            Filtered tensor of same shape.
        """
        B, C, H, W = img.shape
        padding = self.kernel_size // 2

        # Pad the input
        img_padded = F.pad(img, (padding, padding, padding, padding), mode="reflect")

        # Extract patches using unfold
        patches = img_padded.unfold(2, self.kernel_size, 1).unfold(
            3, self.kernel_size, 1
        )
        # patches shape: [B, C, H, W, kernel_size, kernel_size]

        # Get center pixel values
        center = img.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]

        # Compute intensity/color weights (Gaussian on intensity difference)
        intensity_diff = patches - center
        color_weights = torch.exp(-0.5 * (intensity_diff / self.sigma_color) ** 2)

        # Get spatial weights (pre-computed Gaussian kernel)
        spatial_weights = self.gaussian_kernel.to(img.dtype).view(
            1, 1, 1, 1, self.kernel_size, self.kernel_size
        )

        # Combined weights
        weights = color_weights * spatial_weights

        # Normalize weights
        weights_sum = weights.sum(dim=[-2, -1], keepdim=True) + 1e-8
        weights_normalized = weights / weights_sum

        # Apply weighted average
        img_filtered = (patches * weights_normalized).sum(dim=[-2, -1])

        return img_filtered

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create a Gaussian kernel."""
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()

        # Create 1D Gaussian kernel
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        return kernel_2d

    def reverse(self, img):
        """Reverse denoising (identity).

        Note: Denoising is a lossy operation that cannot be reversed.
        This returns the input unchanged.

        Args:
            img (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            img (torch.Tensor): Input unchanged.
        """
        # Denoising cannot be reversed; return input as-is
        return img

"""Denoise.

Reference:
    [1] It is important to remove sensor noise before applying demosaic/CFA, otherwise the CFA will produce zipper noise which is harder to remove. "Thus, it is desirable to suppress the zipper noise in the interpolation stage instead of using noise reduction filter after color interpolation", page 31, Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2.
    [2] Denoise can also be implemented with deep learning methods, replacing the classical denoise filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoise(nn.Module):
    """Noise reduction."""
    
    def __init__(self, method="gaussian", kernel_size=3, sigma=0.5):
        """Initialize denoise.
        
        Args:
            method: Noise reduction method, "gaussian" or "median".
            kernel_size: Size of the kernel.
            sigma: Standard deviation for Gaussian kernel.
        """
        super().__init__()
        self.method = method
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
    
    def forward(self, img):
        """Apply denoise.
        
        Args:
            img (torch.Tensor): Input tensor of shape [B, C, H, W], data range [0, 1].
            
        Returns:
            img_filtered (torch.Tensor): Denoised image, data range [0, 1].
        """
        if self.method == "gaussian":
            # Create Gaussian kernel
            kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma, img.device)
            
            # Apply Gaussian filter
            padding = self.kernel_size // 2
            img_filtered = F.conv2d(
                img, 
                kernel.expand(img.shape[1], 1, self.kernel_size, self.kernel_size), 
                padding=padding,
                groups=img.shape[1]
            )
            
        elif self.method == "median":
            # Apply median filter
            padding = self.kernel_size // 2
            B, C, H, W = img.shape
            img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
            img_filtered = torch.zeros_like(img)
            
            for b in range(B):
                for c in range(C):
                    for i in range(H):
                        for j in range(W):
                            patch = img_padded[b, c, i:i+self.kernel_size, j:j+self.kernel_size]
                            img_filtered[b, c, i, j] = torch.median(patch)
        
        elif self.method is None:
            # No denoising
            img_filtered = img
        
        else:
            raise ValueError(f"Unknown noise reduction method: {self.method}")
            
        return img_filtered
    
    def _create_gaussian_kernel(self, kernel_size, sigma, device):
        """Create a Gaussian kernel."""
        x = torch.arange(kernel_size, device=device) - kernel_size // 2
        x = x.float()
        
        # Create 1D Gaussian kernel
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        return kernel_2d 
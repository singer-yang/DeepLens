"""Dead pixel correction (DPC)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeadPixelCorrection(nn.Module):
    """Dead pixel correction (DPC)."""
    
    def __init__(self, threshold=30, kernel_size=3):
        """Initialize dead pixel correction.
        
        Args:
            threshold: Threshold for detecting dead pixels.
            kernel_size: Size of the kernel for correction.
        """
        super().__init__()
        self.threshold = threshold
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    def forward(self, bayer_nbit):
        """Dead Pixel Correction.
        
        Args:
            bayer_nbit (torch.Tensor): Input n-bit bayer image [B, 1, H, W].
            
        Returns:
            bayer_corrected (torch.Tensor): Corrected n-bit bayer image [B, 1, H, W].

        Reference:
            [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/dpc.py
        """
        padding = self.kernel_size // 2
        
        # Pad the input
        bayer_padded = F.pad(bayer_nbit, (padding, padding, padding, padding), mode='reflect')
        
        # Extract center pixels
        center = bayer_nbit
        
        # Create a median filter
        B, C, H, W = bayer_nbit.shape
        corrected = torch.zeros_like(center)
        
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    patch = bayer_padded[b, 0, i:i+self.kernel_size, j:j+self.kernel_size]
                    corrected[b, 0, i, j] = torch.median(patch)
        
        # Detect dead pixels (pixels that differ significantly from their neighbors)
        diff = torch.abs(center - corrected)
        mask = diff > self.threshold
        
        # Combine original and corrected values using mask
        result = torch.where(mask, corrected, center)
        
        return result.to(torch.uint16) 
# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

import torch
import torch.nn as nn


class PSFLoss(nn.Module):
    def __init__(self, w_achromatic=1.0, w_psf_size=1.0):
        super(PSFLoss, self).__init__()
        self.w_achromatic = w_achromatic
        self.w_psf_size = w_psf_size

    def forward(self, psf):
        # Ensure psf has shape [batch, channels, height, width]
        if psf.dim() == 3:
            psf = psf.unsqueeze(0)  # Add batch dimension
        elif psf.dim() == 2:
            psf = (
                psf.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            )  # Add batch and channel dimensions

        batch, channels, height, width = psf.shape

        # Normalize PSF across spatial dimensions
        psf_normalized = psf / psf.view(batch, channels, -1).sum(
            dim=2, keepdim=True
        ).view(batch, channels, 1, 1)

        # Concentration Loss: Minimize the spatial variance
        # Compute coordinates
        x = torch.linspace(-1, 1, steps=width, device=psf.device, dtype=torch.float32)
        y = torch.linspace(-1, 1, steps=height, device=psf.device, dtype=torch.float32)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv = xv.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        yv = yv.unsqueeze(0).unsqueeze(0)

        # Calculate mean positions
        mean_x = (psf_normalized * xv).sum(dim=(2, 3))
        mean_y = (psf_normalized * yv).sum(dim=(2, 3))

        # Calculate variance
        var_x = ((xv - mean_x.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        var_y = ((yv - mean_y.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        concentration_loss = var_x + var_y
        concentration_loss = concentration_loss.mean()

        # Achromatic Loss: Minimize differences between channels
        channel_diff = 0
        for i in range(channels):
            for j in range(i + 1, channels):
                channel_diff += torch.mean((psf[:, i, :, :] - psf[:, j, :, :]) ** 2)
        channel_diff = channel_diff / (channels * (channels - 1) / 2)

        total_loss = (
            self.w_psf_size * concentration_loss + self.w_achromatic * channel_diff
        )
        return total_loss

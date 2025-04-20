"""Demosaic, or Color Filter Array (CFA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(nn.Module):
    """Demosaic, or Color Filter Array (CFA)."""

    def __init__(self, bayer_pattern="rggb", method="bilinear"):
        """Initialize demosaic.

        Args:
            bayer_pattern: Bayer pattern, "rggb" or "bggr".
            method: Demosaic method, "bilinear" or "3x3".
        """
        super().__init__()
        self.bayer_pattern = bayer_pattern
        self.method = method

    def _bilinear_demosaic(self, bayer):
        """Bilinear interpolation demosaic method.

        Args:
            bayer (torch.Tensor): Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            raw_rgb (torch.Tensor): Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        raise Exception("Bilinear demosaic has a bug.")
        B, C, H, W = bayer.shape
        raw_rgb = torch.zeros((B, 3, H, W), device=bayer.device, dtype=bayer.dtype)

        # Pad the bayer image for boundary handling
        bayer_pad = F.pad(bayer, (1, 1, 1, 1), mode="reflect")

        # Red component
        raw_rgb[:, 0, 0::2, 0::2] = bayer[:, 0, 0::2, 0::2]  # R at (0,0)
        raw_rgb[:, 0, 0::2, 1::2] = (
            bayer_pad[:, 0, 1:-1:2, 2::2] + bayer_pad[:, 0, 1:-1:2, :-2:2]
        ) / 2  # R at (0,1)
        raw_rgb[:, 0, 1::2, 0::2] = (
            bayer_pad[:, 0, 2::2, 1:-1:2] + bayer_pad[:, 0, :-2:2, 1:-1:2]
        ) / 2  # R at (1,0)
        raw_rgb[:, 0, 1::2, 1::2] = (
            bayer_pad[:, 0, 2::2, 2::2]
            + bayer_pad[:, 0, :-2:2, :-2:2]
            + bayer_pad[:, 0, 2::2, :-2:2]
            + bayer_pad[:, 0, :-2:2, 2::2]
        ) / 4  # R at (1,1)

        # Green component
        raw_rgb[:, 1, 0::2, 1::2] = bayer[:, 0, 0::2, 1::2]  # G at (0,1)
        raw_rgb[:, 1, 1::2, 0::2] = bayer[:, 0, 1::2, 0::2]  # G at (1,0)
        raw_rgb[:, 1, 0::2, 0::2] = (
            bayer_pad[:, 0, 1:-1:2, 1:-1:2]
            + bayer_pad[:, 0, 1:-1:2, 1:-1:2]
            + bayer_pad[:, 0, 1:-1:2, 1:-1:2]
            + bayer_pad[:, 0, 1:-1:2, 1:-1:2]
        ) / 4  # G at (0,0)
        raw_rgb[:, 1, 1::2, 1::2] = (
            bayer_pad[:, 0, 2::2, 1:-1:2]
            + bayer_pad[:, 0, :-2:2, 1:-1:2]
            + bayer_pad[:, 0, 1:-1:2, 2::2]
            + bayer_pad[:, 0, 1:-1:2, :-2:2]
        ) / 4  # G at (1,1)

        # Blue component
        raw_rgb[:, 2, 1::2, 1::2] = bayer[:, 0, 1::2, 1::2]  # B at (1,1)
        raw_rgb[:, 2, 0::2, 1::2] = (
            bayer_pad[:, 0, 0:-2:2, 2::2] + bayer_pad[:, 0, 2::2, 2::2]
        ) / 2  # B at (0,1)
        raw_rgb[:, 2, 1::2, 0::2] = (
            bayer_pad[:, 0, 2::2, 0:-2:2] + bayer_pad[:, 0, 2::2, 2::2]
        ) / 2  # B at (1,0)
        raw_rgb[:, 2, 0::2, 0::2] = (
            bayer_pad[:, 0, 0:-2:2, 0:-2:2]
            + bayer_pad[:, 0, 0:-2:2, 2::2]
            + bayer_pad[:, 0, 2::2, 0:-2:2]
            + bayer_pad[:, 0, 2::2, 2::2]
        ) / 4  # B at (0,0)

        return raw_rgb

    def _3x3_demosaic(self, bayer):
        """3x3 kernel-based demosaic method.

        Args:
            bayer: Input tensor of shape [B, 1, H, W].

        Returns:
            raw_rgb: Output tensor of shape [B, 3, H, W].
        """
        B, C, H, W = bayer.shape
        raw_rgb = torch.zeros((B, 3, H, W), device=bayer.device, dtype=bayer.dtype)

        # Create masks for red, green, and blue pixels according to RGGB pattern
        red_mask = torch.zeros_like(bayer)
        green_mask = torch.zeros_like(bayer)
        blue_mask = torch.zeros_like(bayer)

        # Red pixel mask (R) - top-left pixel of the 2x2 block
        red_mask[:, :, 0::2, 0::2] = 1

        # Green pixel masks (G) - top-right and bottom-left pixels of the 2x2 block
        green_mask[:, :, 0::2, 1::2] = 1  # Top-right green
        green_mask[:, :, 1::2, 0::2] = 1  # Bottom-left green

        # Blue pixel mask (B) - bottom-right pixel of the 2x2 block
        blue_mask[:, :, 1::2, 1::2] = 1

        # Extract known color values
        raw_rgb[:, 0, :, :] = (bayer * red_mask).squeeze(1)  # Red channel
        raw_rgb[:, 1, :, :] = (bayer * green_mask).squeeze(1)  # Green channel
        raw_rgb[:, 2, :, :] = (bayer * blue_mask).squeeze(1)  # Blue channel

        # Define interpolation kernels
        kernel_G = (
            torch.tensor(
                [[0, 1 / 4, 0], [1 / 4, 1, 1 / 4], [0, 1 / 4, 0]],
                dtype=bayer.dtype,
                device=bayer.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kernel_RB = (
            torch.tensor(
                [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]],
                dtype=bayer.dtype,
                device=bayer.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Interpolate green channel
        mask_G = (green_mask > 0).float()
        raw_rgb_G = raw_rgb[:, 1, :, :].unsqueeze(1)
        raw_rgb[:, 1, :, :] = (
            F.conv2d(raw_rgb_G * mask_G, kernel_G, padding=1)
            / F.conv2d(mask_G, kernel_G, padding=1)
        ).squeeze()

        # Interpolate red channel
        mask_R = (red_mask > 0).float()
        raw_rgb_R = raw_rgb[:, 0, :, :].unsqueeze(1)
        raw_rgb[:, 0, :, :] = (
            F.conv2d(raw_rgb_R * mask_R, kernel_RB, padding=1)
            / F.conv2d(mask_R, kernel_RB, padding=1)
        ).squeeze()

        # Interpolate blue channel
        mask_B = (blue_mask > 0).float()
        raw_rgb_B = raw_rgb[:, 2, :, :].unsqueeze(1)
        raw_rgb[:, 2, :, :] = (
            F.conv2d(raw_rgb_B * mask_B, kernel_RB, padding=1)
            / F.conv2d(mask_B, kernel_RB, padding=1)
        ).squeeze()

        return raw_rgb

    def forward(self, bayer):
        """Demosaic a Bayer pattern image to RGB.

        Args:
            bayer: Input tensor of shape [B, 1, H, W].

        Returns:
            rgb: Output tensor of shape [B, 3, H, W].
        """
        if bayer.dim() == 3:
            bayer = bayer.unsqueeze(0)
            batch_dim = False
        else:
            batch_dim = True

        if self.method == "bilinear":
            raw_rgb = self._bilinear_demosaic(bayer)
        elif self.method == "3x3":
            raw_rgb = self._3x3_demosaic(bayer)
        else:
            raise ValueError(f"Invalid demosaic method: {self.method}")

        if not batch_dim:
            raw_rgb = raw_rgb.squeeze(0)

        return raw_rgb

    def reverse(self, img):
        """Inverse demosaic from RAW RGB to RAW Bayer.

        Args:
            img (torch.Tensor): RAW RGB image, shape [3, H, W] or [B, 3, H, W], data range [0, 1].

        Returns:
            torch.Tensor: Bayer image, shape [1, H, W] or [B, 1, H, W], data range [0, 1].
        """
        if img.ndim == 3:
            # Input shape: [3, H, W]
            batch_dim = False
            C, H, W = img.shape
        elif img.ndim == 4:
            # Input shape: [B, 3, H, W]
            batch_dim = True
            B, C, H, W = img.shape
        else:
            raise ValueError(
                "Input image must have 3 or 4 dimensions corresponding to [3, H, W] or [B, 3, H, W]."
            )

        if C != 3:
            raise ValueError("Input image must have 3 channels corresponding to RGB.")

        if batch_dim:
            bayer = torch.zeros((B, 1, H, W), dtype=img.dtype, device=img.device)
            bayer[:, 0, 0::2, 0::2] = img[:, 0, 0::2, 0::2]
            bayer[:, 0, 0::2, 1::2] = img[:, 1, 0::2, 1::2]
            bayer[:, 0, 1::2, 0::2] = img[:, 1, 1::2, 0::2]
            bayer[:, 0, 1::2, 1::2] = img[:, 2, 1::2, 1::2]
        else:
            bayer = torch.zeros((1, H, W), dtype=img.dtype, device=img.device)
            bayer[0, 0::2, 0::2] = img[0, 0::2, 0::2]
            bayer[0, 0::2, 1::2] = img[1, 0::2, 1::2]
            bayer[0, 1::2, 0::2] = img[1, 1::2, 0::2]
            bayer[0, 1::2, 1::2] = img[2, 1::2, 1::2]

        return bayer

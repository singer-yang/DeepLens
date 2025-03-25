"""Auto white balance (AWB)."""

import random

import torch
import torch.nn as nn


class AutoWhiteBalance(nn.Module):
    """Auto white balance (AWB)."""

    def __init__(self, awb_method="gray_world", gains=(2.0, 1.0, 1.8)):
        """Initialize auto white balance.

        Args:
            awb_method: AWB method, "gray_world" or "manual".
            gains: RGB gains for manual AWB, shape [3].
        """
        super().__init__()
        self.awb_method = awb_method
        self.register_buffer('gains', torch.tensor(gains))

    def sample_augmentation(self):
        if not hasattr(self, "gains_org"):
            self.gains_org = self.gains
        self.gains = self.gains_org + torch.randn_like(self.gains_org) * 0.1

    def reset_augmentation(self):
        self.gains = self.gains_org

    def apply_awb_bayer(self, bayer):
        """Apply white balance to Bayer pattern image.

        Args:
            bayer: Input tensor of shape [B, 1, H, W].

        Returns:
            bayer_wb: Output tensor with same shape as input.
        """
        B, _, H, W = bayer.shape

        # Create masks for R, G, B pixels (assuming RGGB pattern)
        r_mask = torch.zeros((H, W), device=bayer.device)
        g_mask = torch.zeros((H, W), device=bayer.device)
        b_mask = torch.zeros((H, W), device=bayer.device)

        r_mask[0::2, 0::2] = 1  # R at top-left
        g_mask[0::2, 1::2] = 1  # G at top-right
        g_mask[1::2, 0::2] = 1  # G at bottom-left
        b_mask[1::2, 1::2] = 1  # B at bottom-right

        # Apply masks to extract color channels
        r = bayer * r_mask.view(1, 1, H, W)
        g = bayer * g_mask.view(1, 1, H, W)
        b = bayer * b_mask.view(1, 1, H, W)

        if self.awb_method == "gray_world":
            # Calculate average for each channel (excluding zeros)
            r_avg = torch.sum(r, dim=[2, 3]) / torch.sum(r_mask)
            g_avg = torch.sum(g, dim=[2, 3]) / torch.sum(g_mask)
            b_avg = torch.sum(b, dim=[2, 3]) / torch.sum(b_mask)

            # Calculate gains to make averages equal
            g_gain = torch.ones_like(g_avg)
            r_gain = g_avg / (r_avg + 1e-6)
            b_gain = g_avg / (b_avg + 1e-6)

            # Apply gains
            bayer_wb = bayer.clone()
            bayer_wb = bayer_wb * (
                r_mask.view(1, 1, H, W) * r_gain.view(B, 1, 1, 1)
                + g_mask.view(1, 1, H, W) * g_gain.view(B, 1, 1, 1)
                + b_mask.view(1, 1, H, W) * b_gain.view(B, 1, 1, 1)
            )

        elif self.awb_method == "manual":
            # Apply manual gains
            bayer_wb = bayer.clone()
            bayer_wb = bayer_wb * (
                r_mask.view(1, 1, H, W) * self.gains[0]
                + g_mask.view(1, 1, H, W) * self.gains[1]
                + b_mask.view(1, 1, H, W) * self.gains[2]
            )
        else:
            raise ValueError(f"Unknown AWB method: {self.awb_method}")

        return bayer_wb

    def apply_awb_rgb(self, rgb):
        """Apply white balance to RGB image.

        Args:
            rgb: Input tensor of shape [B, 3, H, W].

        Returns:
            rgb_wb: Output tensor with same shape as input.
        """
        if self.awb_method == "gray_world":
            # Calculate average for each channel
            rgb_avg = torch.mean(rgb, dim=[2, 3], keepdim=True)

            # Calculate gains to make averages equal
            g_avg = rgb_avg[:, 1:2, :, :]
            gains = g_avg / (rgb_avg + 1e-6)

            # Apply gains
            rgb_wb = rgb * gains

        elif self.awb_method == "manual":
            # Apply manual gains
            rgb_wb = rgb * self.gains.view(1, 3, 1, 1)
        
        else:
            raise ValueError(f"Unknown AWB method: {self.awb_method}")

        return rgb_wb

    def forward(self, input_tensor):
        """Auto White Balance (AWB).

        Args:
            input_tensor: Input tensor of shape [B, 1, H, W] or [B, 3, H, W].

        Returns:
            output_tensor: Output tensor [B, 1, H, W] or [B, 3, H, W].
        """
        if input_tensor.shape[1] == 1:
            return self.apply_awb_bayer(input_tensor)
        else:
            return self.apply_awb_rgb(input_tensor)

    def reverse(self, img):
        """Inverse auto white balance."""
        kr = self.gains[0]
        kg = self.gains[1]
        kb = self.gains[2]

        # Inverse AWB
        rgb_unbalanced = torch.zeros_like(img)
        if len(img.shape) == 3:
            rgb_unbalanced[0, :, :] = img[0, :, :] / kr
            rgb_unbalanced[1, :, :] = img[1, :, :] / kg
            rgb_unbalanced[2, :, :] = img[2, :, :] / kb
        else:
            rgb_unbalanced[:, 0, :, :] = img[:, 0, :, :] / kr
            rgb_unbalanced[:, 1, :, :] = img[:, 1, :, :] / kg
            rgb_unbalanced[:, 2, :, :] = img[:, 2, :, :] / kb

        return rgb_unbalanced

    def safe_reverse_awb(self, img):
        """Inverse auto white balance.

        Ref: https://github.com/google-research/google-research/blob/master/unprocessing/unprocess.py#L92C1-L102C28
        """
        kr = self.gains[0]
        kg = self.gains[1]
        kb = self.gains[2]

        # Safely inverse AWB
        if len(img.shape) == 3:
            gains = (
                torch.tensor([1.0 / kr, 1.0 / kg, 1.0 / kb], device=img.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            gray = torch.mean(img, dim=0, keepdim=True)
            inflection = 0.9
            mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)

            rgb_unbalanced = img * safe_gains

        elif len(img.shape) == 4:
            gains = (
                torch.tensor([1.0 / kr, 1.0 / kg, 1.0 / kb], device=rgb.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
            )

            gray = torch.mean(img, dim=1, keepdim=True)
            inflection = 0.9
            mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)

            rgb_unbalanced = img * safe_gains

        else:
            raise ValueError("Invalid rgb shape")

        return rgb_unbalanced

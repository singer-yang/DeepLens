"""Anti-aliasing filter (AAF)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasingFilter(nn.Module):
    """Anti-Aliasing Filter (AAF)."""

    def __init__(self, method="bilateral"):
        """Initialize the Anti-Aliasing Filter.

        Args:
            method (str): Denoising method to use. Options: "bilateral", "none", or None.
                          If "none" or None, no filtering is applied.
        """
        super(AntiAliasingFilter, self).__init__()
        self.method = method

    def forward(self, bayer_nbit):
        """Apply anti-aliasing filter to remove moiré pattern.

        Args:
            bayer_nbit: Input tensor of shape [B, 1, H, W], data range [0, 1]

        Returns:
            Filtered bayer tensor of same shape as input

        Reference:
            [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/aaf.py
        """
        raise NotImplementedError("Anti-aliasing filter is not tested yet.")
        if self.method is None or self.method == "none":
            return bayer_nbit

        elif self.method == "bilateral":
            # Convert to int32 for calculations
            bayer = bayer_nbit.to(torch.int32)

            # Pad the input with reflection padding
            padded = F.pad(bayer, (2, 2, 2, 2), mode="reflect")

            # Get all 9 shifted versions for 3x3 window
            shifts = []
            for i in range(3):
                for j in range(3):
                    shifts.append(
                        padded[:, :, i : i + bayer.shape[2], j : j + bayer.shape[3]]
                    )

            # Initialize result tensor
            result = torch.zeros_like(shifts[0], dtype=torch.int32)

            # Apply weights: center pixel (index 4) gets weight 8, others get weight 1
            for i, shifted in enumerate(shifts):
                weight = 8 if i == 4 else 1
                result += weight * shifted

            # Right shift by 4 (divide by 16)
            result = result >> 4

            return result.to(torch.uint16)

        else:
            raise ValueError(f"Unknown denoise method: {self.method}")

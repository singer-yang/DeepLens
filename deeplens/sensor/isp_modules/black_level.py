"""Black level compensation (BLC)."""

import torch
import torch.nn as nn


class BlackLevelCompensation(nn.Module):
    """Black level compensation (BLC)."""

    def __init__(self, bit=10, black_level=64):
        """Initialize black level compensation.

        Args:
            bit: Bit depth of the input image.
            black_level: Black level value.
        """
        super().__init__()
        self.bit = bit
        self.black_level = black_level

    def forward(self, bayer):
        """Black Level Compensation.

        Args:
            bayer (torch.Tensor): Input n-bit bayer image [B, 1, H, W], data range [~black_level, 2**bit - 1].

        Returns:
            bayer_float (torch.Tensor): Output float bayer image [B, 1, H, W], data range [0, 1].
        """
        # Subtract black level
        bayer_float = (bayer - self.black_level) / (2**self.bit - 1 - self.black_level)

        # Clamp to [0, 1], (unnecessary)
        bayer_float = torch.clamp(bayer_float, 0.0, 1.0)

        return bayer_float

    def reverse(self, bayer):
        """Inverse black level compensation.

        Args:
            bayer: Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            bayer_nbit: Output tensor of shape [B, 1, H, W], data range [0, 2^bit-1].
        """
        max_value = 2**self.bit - 1
        bayer_nbit = bayer * (max_value - self.black_level) + self.black_level
        bayer_nbit = torch.round(bayer_nbit)
        return bayer_nbit

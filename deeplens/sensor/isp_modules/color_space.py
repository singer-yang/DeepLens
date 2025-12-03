"""Color space conversion (CSC)."""

import torch
import torch.nn as nn


class ColorSpaceConversion(nn.Module):
    """Color space conversion (CSC).

    Color space conversion is a technique to convert the color space of the image.
    """

    def __init__(self):
        """Initialize color space conversion module."""
        super().__init__()

        # RGB to YCrCb conversion matrix
        self.register_buffer(
            "rgb_to_ycrcb_matrix",
            torch.tensor(
                [
                    [0.299, 0.587, 0.114],
                    [0.5, -0.4187, -0.0813],
                    [-0.1687, -0.3313, 0.5],
                ]
            ),
        )

        # YCrCb to RGB conversion matrix
        self.register_buffer(
            "ycrcb_to_rgb_matrix",
            torch.tensor(
                [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]]
            ),
        )

    def rgb_to_ycrcb(self, rgb_image):
        """Convert RGB to YCrCb (differentiable).

        Args:
            rgb_image: Input tensor of shape [B, 3, H, W] in RGB format.

        Returns:
            ycrcb_image: Output tensor of shape [B, 3, H, W] in YCrCb format.

        Reference:
            [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/csc.py
        """
        # Reshape for matrix multiplication
        rgb_reshaped = rgb_image.permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Apply transformation
        ycrcb = torch.matmul(rgb_reshaped, self.rgb_to_ycrcb_matrix.T)

        # Add offset to Cr and Cb (non-in-place for gradient flow)
        offset = torch.tensor([0.0, 0.5, 0.5], device=ycrcb.device, dtype=ycrcb.dtype)
        ycrcb = ycrcb + offset

        # Reshape back
        ycrcb_image = ycrcb.permute(0, 3, 1, 2)  # [B, 3, H, W]

        return ycrcb_image

    def ycrcb_to_rgb(self, ycrcb_image):
        """Convert YCrCb to RGB (differentiable).

        Args:
            ycrcb_image: Input tensor of shape [B, 3, H, W] in YCrCb format.

        Returns:
            rgb_image: Output tensor of shape [B, 3, H, W] in RGB format.
        """
        # Reshape for matrix multiplication
        ycrcb = ycrcb_image.permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Subtract offset from Cr and Cb (non-in-place for gradient flow)
        offset = torch.tensor([0.0, 0.5, 0.5], device=ycrcb.device, dtype=ycrcb.dtype)
        ycrcb_adj = ycrcb - offset

        # Apply transformation
        rgb = torch.matmul(ycrcb_adj, self.ycrcb_to_rgb_matrix.T)

        # Clamp values to [0, 1]
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # Reshape back
        rgb_image = rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

        return rgb_image

    def forward(self, image, conversion="rgb_to_ycrcb"):
        """Convert between color spaces.

        Args:
            image: Input tensor of shape [B, 3, H, W].
            conversion: Conversion direction, "rgb_to_ycrcb" or "ycrcb_to_rgb".

        Returns:
            converted_image: Output tensor of shape [B, 3, H, W].
        """
        if conversion == "rgb_to_ycrcb":
            return self.rgb_to_ycrcb(image)
        elif conversion == "ycrcb_to_rgb":
            return self.ycrcb_to_rgb(image)
        else:
            raise ValueError(f"Unknown conversion: {conversion}")

    def reverse(self, image):
        """Reverse color space conversion (YCrCb to RGB).

        This is the inverse of the forward pass (which defaults to rgb_to_ycrcb).

        Args:
            image: Input tensor of shape [B, 3, H, W] in YCrCb format.

        Returns:
            rgb_image: Output tensor of shape [B, 3, H, W] in RGB format.
        """
        return self.ycrcb_to_rgb(image)

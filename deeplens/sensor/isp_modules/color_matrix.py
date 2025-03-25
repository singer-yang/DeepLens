"""Color correction matrix (CCM)."""

import torch
import torch.nn as nn

class ColorCorrectionMatrix(nn.Module):
    """Color correction matrix (CCM)."""
    
    def __init__(self, ccm_matrix=None):
        """Initialize color correction matrix.
        
        Args:
            ccm_matrix: Color correction matrix of shape [4, 3].

        Reference:
            [1] https://github.com/QiuJueqin/fast-openISP/blob/master/configs/nikon_d3200.yaml#L57
            [2] https://github.com/timothybrooks/hdr-plus/blob/master/src/finish.cpp#L626
            ccm_matrix = torch.tensor(
                [
                    [1.8506, -0.7920, -0.0605],
                    [-0.1562, 1.6455, -0.4912],
                    [0.0176, -0.5439, 1.5254],
                    [0.0, 0.0, 0.0],
                ]
            )
        """
        super().__init__()
        if ccm_matrix is None:
            ccm_matrix = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]
            ])
        self.register_buffer('ccm_matrix', ccm_matrix)

    def sample_augmentation(self):
        if not hasattr(self, "ccm_org"):
            self.ccm_org = self.ccm_matrix
        self.ccm_matrix = self.ccm_org + torch.randn_like(self.ccm_org) * 0.01

    def reset_augmentation(self):
        self.ccm_matrix = self.ccm_org
    
    def forward(self, rgb_image):
        """Color Correction Matrix. Convert RGB image to sensor color space.

        Args:
            rgb_image: Input tensor of shape [B, 3, H, W] in RGB format.

        Returns:
            rgb_corrected: Corrected RGB image in sensor color space.
        """
        # Extract matrix and bias
        matrix = self.ccm_matrix[:3, :]  # Shape: (3, 3)
        bias = self.ccm_matrix[3, :].view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)

        # Apply CCM
        # Reshape rgb_image to [B, H, W, 3] for matrix multiplication
        rgb_image_perm = rgb_image.permute(0, 2, 3, 1)  # [B, H, W, 3]
        rgb_corrected = torch.matmul(rgb_image_perm, matrix.T) + bias.squeeze()
        rgb_corrected = rgb_corrected.permute(0, 3, 1, 2)  # [B, 3, H, W]

        return rgb_corrected 
    
    def reverse(self, img):
        """Inverse color correction matrix. Convert sensor color space to RGB image.

        Args:
            rgb_image: Input tensor of shape [B, 3, H, W] in sensor color space.
        """
        ccm_matrix = self.ccm_matrix

        # Extract matrix and bias from CCM
        matrix = ccm_matrix[:3, :]  # Shape: (3, 3)
        bias = ccm_matrix[3, :].view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)

        # Compute the inverse of the CCM matrix
        inv_matrix = torch.inverse(matrix)  # Shape: (3, 3)

        # Prepare rgb_corrected for matrix multiplication
        img_perm = img.permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Subtract bias
        img_minus_bias = img_perm - bias.squeeze()

        # Apply Inverse CCM
        img_original = torch.matmul(img_minus_bias, inv_matrix.T)  # [B, H, W, 3]
        img_original = img_original.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Clip the values to ensure they are within the valid range
        img_original = torch.clamp(img_original, 0.0, 1.0)

        return img_original
"""Gamma correction (GC)."""

import random

import torch
import torch.nn as nn


class GammaCorrection(nn.Module):
    """Gamma correction (GC).
    
    Gamma correction is a technique to adjust the gamma of the image.
    """

    def __init__(self, gamma_param=2.2):
        """Initialize gamma correction module.

        Args:
            gamma_param: Gamma parameter. Default is 2.2.
        """
        super().__init__()
        self.register_buffer("gamma_param", torch.tensor(gamma_param))

    def sample_augmentation(self):
        """Sample augmentation for synthetic data generation."""
        if not hasattr(self, "gamma_param_org"):
            self.gamma_param_org = self.gamma_param
        self.gamma_param = (
            self.gamma_param_org + torch.randn_like(self.gamma_param_org) * 0.01
        )

    def reset_augmentation(self):
        """Reset augmentation for evaluation."""
        self.gamma_param = self.gamma_param_org

    def forward(self, img, quantize=False):
        """Gamma Correction (differentiable).

        Args:
            img (tensor): Input image. Shape of [B, C, H, W].
            quantize (bool): Whether to quantize the image to 8-bit.
                             WARNING: quantize=True makes this non-differentiable!

        Returns:
            img_gamma (tensor): Gamma corrected image. Shape of [B, C, H, W].

        Reference:
            [1] "There is no restriction as to where stage gamma correction is placed," page 35, Architectural Analysis of a Baseline ISP Pipeline.
        """
        img_gamma = torch.pow(torch.clamp(img, min=1e-8), 1 / self.gamma_param)
        if quantize:
            # WARNING: torch.round() is NOT differentiable!
            # Use only for final output, not during training
            img_gamma = torch.round(img_gamma * 255) / 255
        return img_gamma

    def reverse(self, img):
        """Inverse gamma correction.

        Args:
            img (tensor): Input image. Shape of [B, C, H, W].

        Returns:
            img (tensor): Inverse gamma corrected image. Shape of [B, C, H, W].

        Reference:
            [1] https://github.com/google-research/google-research/blob/master/unprocessing/unprocess.py#L78
        """
        gamma_param = self.gamma_param
        img = torch.clip(img, 1e-8) ** gamma_param
        return img

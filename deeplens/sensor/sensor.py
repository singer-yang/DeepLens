"""Minimal base sensor class.

This provides the simplest sensor model: size, resolution, and gamma-only ISP.
For sensors with noise models and bit depth, use MonoSensor or RGBSensor.

Note: This sensor model is used in various renderers, including Blender, for 
physically-based camera simulation.
"""

import torch.nn as nn

from deeplens.sensor.isp_modules.gamma_correction import GammaCorrection


class Sensor(nn.Module):
    def __init__(self, size=(8.0, 6.0), res=(4000, 3000)):
        super().__init__()

        # Sensor size and resolution
        self.size = size
        self.res = res

        # ISP: gamma correction only
        self.isp = nn.Sequential(
            GammaCorrection(),
        )

    def to(self, device):
        self.device = device
        self.isp.to(device)
        return self

    def forward(self, img):
        """Apply gamma correction to a linear image.

        Args:
            img: Tensor of shape (B, C, H, W), range [0, 1]

        Returns:
            img_out: Tensor of shape (B, C, H, W), range [0, 1]
        """
        img_out = self.simu_noise(img)
        img_out = self.response_curve(img_out)
        img_out = self.isp(img_out)
        return img_out

    def response_curve(self, img_irr):
        """Apply response curve to the irradiance image to get the raw image.

        Default is identity (linear response).

        Args:
            img_irr: Irradiance image

        Returns:
            img_raw: Raw image
        """
        return img_irr

    def simu_noise(self, img):
        """Simulate sensor noise.

        Default is identity (no noise).

        Args:
            img: Input image

        Returns:
            img: Same image unchanged
        """
        return img

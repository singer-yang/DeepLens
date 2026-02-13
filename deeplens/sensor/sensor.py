"""Minimal base sensor class.

This provides the simplest sensor model: size, resolution, and gamma-only ISP.
For sensors with noise models and bit depth, use MonoSensor or RGBSensor.

Note: This sensor model is used in various renderers, including Blender, for
physically-based camera simulation.
"""

import json

import torch.nn as nn

from deeplens.sensor.isp_modules.gamma_correction import GammaCorrection


class Sensor(nn.Module):
    def __init__(self, size=(8.0, 6.0), res=(4000, 3000)):
        """
        Args:
            size (tuple): Sensor physical size in mm (W, H). Default (8.0, 6.0).
            res (tuple): Sensor resolution in pixels (W, H). Default (4000, 3000).
        """
        super().__init__()

        # Sensor size and resolution
        self.size = size
        self.res = res

        # ISP: gamma correction only
        self.isp = nn.Sequential(
            GammaCorrection(),
        )

    @classmethod
    def from_config(cls, sensor_file):
        """Create a Sensor from a JSON config file.

        Args:
            sensor_file: Path to JSON sensor config file.

        Returns:
            Sensor instance.
        """
        with open(sensor_file, "r") as f:
            config = json.load(f)

        return cls(
            size=config.get("sensor_size", (8.0, 6.0)),
            res=config.get("sensor_res", (4000, 3000)),
        )

    def to(self, device):
        self.device = device
        self.isp.to(device)
        return self

    def response_curve(self, img_irr):
        """Apply response curve to the irradiance image to get the raw image.

        Default is identity (linear response).

        Args:
            img_irr: Irradiance image

        Returns:
            img_raw: Raw image
        """
        return img_irr

    def unprocess(self, img):
        """Inverse ISP: convert sRGB image back to linear RGB.

        Args:
            img: Tensor of shape (B, C, H, W), range [0, 1] in sRGB space.

        Returns:
            img_linear: Tensor of shape (B, C, H, W), range [0, 1] in linear space.
        """
        # Inverse gamma correction (isp[0] is GammaCorrection)
        return self.isp[0].reverse(img)

    def linear_rgb2raw(self, img_linear):
        """Convert linear RGB image to raw sensor space.

        For the base Sensor, raw is the linear image itself (identity).

        Args:
            img_linear: Tensor of shape (B, C, H, W), range [0, 1].

        Returns:
            img_raw: Tensor of shape (B, C, H, W), range [0, 1].
        """
        return img_linear

    def simu_noise(self, img):
        """Simulate sensor noise.

        Default is identity (no noise).

        Args:
            img: Input image

        Returns:
            img: Same image unchanged
        """
        return img

"""Image sensor, containing noise model and ISP.

This file contains the following sensor classes:
    1. Sensor
    2. MonoSensor
    3. EventSensor
"""

import json
import math

import torch
import torch.nn as nn

from deeplens.sensor.isp import InvertibleISP
from deeplens.sensor.isp_modules.black_level import BlackLevelCompensation


class Sensor(nn.Module):
    def __init__(
        self,
        bit=10,
        black_level=64,
        size=(8.0, 6.0),
        res=(4000, 3000),
        read_noise_std=0.5,
        shot_noise_std_alpha=0.4,
        shot_noise_std_beta=0.0,
        iso_base=100,
        device=None,
    ):
        super().__init__()

        # Sensor resolution and normalized pixel size
        self.size = size
        self.res = res
        self.pixel_size = 2 / math.sqrt(self.res[0] ** 2 + self.res[1] ** 2)

        self.bit = bit
        self.nbit_max = 2**bit - 1
        self.black_level = black_level

        # Sensor noise statistics (should be measured in n-bit digital value space)
        self.iso_base = iso_base  # base iso at analog gain 1
        self.readnoise_std = read_noise_std
        self.shotnoise_std_alpha = shot_noise_std_alpha
        self.shotnoise_std_beta = shot_noise_std_beta

        # ISP
        self.isp = nn.Sequential(
            BlackLevelCompensation(bit, black_level),
        )

    # @classmethod
    # def from_config(cls, config):
    #     """Create a sensor from a config dictionary."""
    #     bit = config.get("bit", 10)
    #     black_level = config.get("black_level", 64)
    #     res = config.get("res", (4000, 3000))
    #     size = config.get("size", (8.0, 6.0))
    #     read_noise_std = config.get("read_noise_std", 0.5)
    #     shot_noise_std_alpha = config.get("shot_noise_std_alpha", 0.5)
    #     shot_noise_std_beta = config.get("shot_noise_std_beta", 0.0)
    #     iso_base = config.get("iso_base", 100)
    #     return cls(
    #         bit=bit,
    #         black_level=black_level,
    #         res=res,
    #         size=size,
    #         read_noise_std=read_noise_std,
    #         shot_noise_std_alpha=shot_noise_std_alpha,
    #         shot_noise_std_beta=shot_noise_std_beta,
    #         iso_base=iso_base,
    #     )

    def to(self, device):
        self.device = device
        self.isp.to(device)
        return self

    def __call__(self, img_nbit, iso):
        return self.forward(img_nbit, iso)

    def forward(self, img_nbit, iso):
        """Simulate sensor output with noise and ISP.

        Args:
            img_nbit: Tensor of shape (B, 3, H, W), range [~black_level, 2**bit - 1]
            iso: ISO value

        Returns:
            img_noisy: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        img_noisy = self.simu_noise(img_nbit, iso)
        img_noisy = self.isp(img_noisy)
        return img_noisy

    def forward_irr(self, img_irr, iso):
        """Simulate sensor output from irradiance field. Not used but kept for reference.

        The optical simulation (PSF, optical aberrations) happens in the irradiance space. But since the sensor response is a linear function, we can directly apply optical simulation to the raw image. That means: response(optics(irr)) = optics(response(irr)) = optics(img_raw)

        Args:
            img_irr: Irradiance image
            iso: ISO value

        Returns:
            img_noisy: Processed image with noise
        """
        img_raw = self.response_curve(img_irr)
        img_noisy = self.simu_noise(img_raw, iso)
        img_noisy = self.isp(img_noisy)
        return img_noisy

    def response_curve(self, img_irr):
        """Apply response curve to the irradiance image to get the raw image.

        Args:
            img_irr: Irradiance image

        Returns:
            img_raw: Raw image
        """
        img_raw = img_irr
        return img_raw

    def simu_noise(self, img_raw, iso):
        """Simulate sensor noise considering sensor quantization and noise model.

        Args:
            img_raw: N-bit clean image, (B, C, H, W), range [0, 2**bit - 1]
            iso: (B,), range [0, 400]

        Returns:
            img_raw_noise: N-bit noisy image, (B, C, H, W), range [0, 2**bit - 1]

        Reference:
            [1] "Unprocessing Images for Learned Raw Denoising."
            [2] https://www.photonstophotos.net/Charts/RN_ADU.htm
            [3] https://www.photonstophotos.net/Investigations/Measurement_and_Sample_Variation.htm
            [4] https://www.dpreview.com/forums/thread/4669806

        Note:
            We can tune the iso channel in the network input to (1) compensate for inaccurate noise model, and (2) achieve better image quality.
        """
        nbit_max = self.nbit_max
        black_level = self.black_level
        device = img_raw.device

        # Calculate noise standard deviation
        shotnoise_std = torch.clamp(
            self.shotnoise_std_alpha * torch.sqrt(torch.clamp(img_raw - black_level, min=0.0))
            + self.shotnoise_std_beta,
            0.0,
        )
        if (iso > 800).any():
            raise ValueError(f"Currently noise model only works for low ISO <= 800, got {iso}")
        gain_analog = 1.0  # we only measured analog gain = 1.0
        gain_digit = (iso / self.iso_base).view(-1, 1, 1, 1)
        noise_std = torch.sqrt(
            shotnoise_std**2 * gain_digit * gain_analog
            + self.readnoise_std**2 * gain_digit**2
        )

        # Sample random noise
        noise_sample = (
            torch.normal(mean=0.0, std=1.0, size=img_raw.size(), device=device)
            * noise_std
        )
        img_raw_noisy = img_raw + noise_sample
        
        # Clip and quantize
        img_raw_noisy = torch.clip(img_raw_noisy, 0.0, nbit_max)
        img_raw_noisy = torch.round(img_raw_noisy)
        return img_raw_noisy



class MonoSensor(Sensor):
    """Monochrome sensor"""

    def __init__(
        self,
        bit=10,
        black_level=64,
        res=(4000, 3000),
        size=(8.0, 6.0),
        iso_base=100,
        read_noise_std=0.5,
        shot_noise_std_alpha=0.4,
        shot_noise_std_beta=0.0,
    ):
        super().__init__(
            bit=bit,
            black_level=black_level,
            res=res,
            size=size,
            iso_base=iso_base,
            read_noise_std=read_noise_std,
            shot_noise_std_alpha=shot_noise_std_alpha,
            shot_noise_std_beta=shot_noise_std_beta,
        )
        self.isp = nn.Sequential(
            BlackLevelCompensation(bit, black_level),
        )

    def forward(self, img_nbit, iso=100.0):
        """Converts light illuminance to monochrome image.

        Args:
            img_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]
            iso: ISO value, default 100.0

        Returns:
            img_noisy: Processed monochrome image with noise
        """
        img_noisy = self.simu_noise(img_nbit, iso)
        img_noisy = self.isp(img_noisy)
        return img_noisy


class EventSensor(Sensor):
    """Event sensor"""

    def __init__(self, bit=10, black_level=64):
        super().__init__(bit, black_level)

    def forward(self, I_t, I_t_1):
        """Converts light illuminance to event stream.

        Args:
            I_t: Current frame
            I_t_1: Previous frame

        Returns:
            Event stream
        """
        # Converts light illuminance to event stream.
        pass

    def forward_video(self, frames):
        """Simulate sensor output from a video.

        Args:
            frames: Tensor of shape (B, T, 3, H, W), range [0, 1]

        Returns:
            Event stream for the video sequence
        """
        pass

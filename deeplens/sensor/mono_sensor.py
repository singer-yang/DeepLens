"""Monochrome sensor with noise model and ISP.

Example: infrared sensor
"""

import torch
import torch.nn as nn

from deeplens.sensor.sensor import Sensor
from deeplens.sensor.isp_modules.black_level import BlackLevelCompensation
from deeplens.sensor.isp_modules.gamma_correction import GammaCorrection


class MonoSensor(Sensor):
    """Monochrome sensor with noise simulation and ISP."""

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
        wavelengths=None,
        spectral_response=None,
    ):
        super().__init__(size=size, res=res)

        self.bit = bit
        self.nbit_max = 2**bit - 1
        self.black_level = black_level

        # Sensor noise statistics (measured in n-bit digital value space)
        self.iso_base = iso_base
        self.readnoise_std = read_noise_std
        self.shotnoise_std_alpha = shot_noise_std_alpha
        self.shotnoise_std_beta = shot_noise_std_beta

        # Spectral response curve
        self.wavelengths = wavelengths
        if self.wavelengths is not None:
            response = torch.tensor(spectral_response, dtype=torch.float32)
            self.spectral_response = response / response.sum()

        # ISP: black level compensation + gamma
        self.isp = nn.Sequential(
            BlackLevelCompensation(bit, black_level),
            GammaCorrection(),
        )

    def to(self, device):
        super().to(device)
        if self.wavelengths is not None:
            self.spectral_response = self.spectral_response.to(device)
        return self

    def response_curve(self, img_spectral):
        """Apply spectral response curve to get a monochrome raw image.

        Args:
            img_spectral: Spectral image, (B, N_wavelengths, H, W)

        Returns:
            img_raw: Monochrome raw image, (B, 1, H, W)
        """
        if self.wavelengths is not None:
            img_raw = (
                img_spectral * self.spectral_response.view(1, -1, 1, 1)
            ).sum(dim=1, keepdim=True)
        else:
            if img_spectral.shape[1] == 1:
                img_raw = img_spectral
            else:
                # Average across channels as fallback
                img_raw = img_spectral.mean(dim=1, keepdim=True)

        return img_raw

    def forward(self, img_nbit, iso):
        """Simulate sensor output with noise and ISP.

        Args:
            img_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]
            iso: ISO value

        Returns:
            img_noisy: Processed monochrome image with noise, range [0, 1]
        """
        img_noisy = self.simu_noise(img_nbit, iso)
        img_noisy = self.isp(img_noisy)
        return img_noisy

    def simu_noise(self, img_raw, iso):
        """Simulate sensor noise considering sensor quantization and noise model.

        Args:
            img_raw: N-bit clean image, (B, C, H, W), range [0, 2**bit - 1]
            iso: (B,), range [0, 800]

        Returns:
            img_raw_noise: N-bit noisy image, (B, C, H, W), range [0, 2**bit - 1]

        Reference:
            [1] "Unprocessing Images for Learned Raw Denoising."
            [2] https://www.photonstophotos.net/Charts/RN_ADU.htm
            [3] https://www.photonstophotos.net/Investigations/Measurement_and_Sample_Variation.htm
            [4] https://www.dpreview.com/forums/thread/4669806
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

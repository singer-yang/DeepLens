import torch.nn as nn

from .sensor import Sensor
from .isp import BlackLevelCompensation

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


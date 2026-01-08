"""RGB sensor with ISP. It is used to convert between RAW bayer image and RGB image."""

import json
import math

import torch

from deeplens.sensor import Sensor
from deeplens.sensor.isp import InvertibleISP


class RGBSensor(Sensor):
    """RGB sensor."""

    def __init__(self, sensor_file):
        super().__init__()
        
        with open(sensor_file, "r") as f:
            config = json.load(f)

        # Extract parameters with defaults
        self.size = config["sensor_size"]
        self.res = config["sensor_res"]
        self.pixel_size = 2 / math.sqrt(self.res[0] ** 2 + self.res[1] ** 2)
        self.bit = config["bit"]
        self.nbit_max = 2**self.bit - 1  
        self.black_level = config["black_level"]
        self.bayer_pattern = config.get("bayer_pattern", "rggb")

        # ISP parameters
        white_balance = config.get("white_balance_d50", (2.0, 1.0, 1.8))
        color_matrix = config.get("color_matrix_d50", None)
        gamma_param = config.get("gamma_param", 2.2)

        # Noise parameters
        self.iso_base = config.get("iso_base", 100)
        self.read_noise_std = config.get("read_noise_std", 0.5)
        self.shot_noise_std_alpha = config.get("shot_noise_std_alpha", 0.4)
        self.shot_noise_std_beta = config.get("shot_noise_std_beta", 0.0)
        
        # Spectral response curves
        self.wavelengths = config.get("wavelengths", None)
        red_response = config.get("red_spectral_response", None)
        green_response = config.get("green_spectral_response", None)
        blue_response = config.get("blue_spectral_response", None)
        if self.wavelengths is not None:
            self.red_response = torch.tensor(red_response) / sum(green_response)
            self.green_response = torch.tensor(green_response) / sum(green_response)
            self.blue_response = torch.tensor(blue_response) / sum(green_response)

        # ISP
        self.isp = InvertibleISP(
            bit=self.bit,
            black_level=self.black_level,
            bayer_pattern=self.bayer_pattern,
            white_balance=white_balance,
            color_matrix=color_matrix,
            gamma_param=gamma_param,
        )

    def to(self, device):
        super().to(device)
        if self.wavelengths is not None:
            self.red_response = self.red_response.to(device)
            self.green_response = self.green_response.to(device)
            self.blue_response = self.blue_response.to(device)
        return self

    def response_curve(self, img_spectral):
        """Apply response curve to the spectral image to get the raw image.

        Args:
            img_spectral: Spectral image

        Returns:
            img_raw: Raw image

        Reference:
            [1] Spectral Sensitivity Estimation Without a Camera. ICCP 2023.
            [2] https://github.com/COLOR-Lab-Eilat/Spectral-sensitivity-estimation
        """
        if self.wavelengths is not None:
            img_raw = torch.zeros(
                (
                    img_spectral.shape[0],
                    3,
                    img_spectral.shape[2],
                    img_spectral.shape[3],
                ),
                device=img_spectral.device,
            )
            img_raw[:, 0, :, :] = (
                img_spectral * self.red_response.view(1, -1, 1, 1)
            ).sum(dim=1)
            img_raw[:, 1, :, :] = (
                img_spectral * self.green_response.view(1, -1, 1, 1)
            ).sum(dim=1)
            img_raw[:, 2, :, :] = (
                img_spectral * self.blue_response.view(1, -1, 1, 1)
            ).sum(dim=1)
        else:
            assert img_spectral.shape[1] == 3, (
                "No spectral response curves provided, input image must have 3 channels"
            )
            img_raw = img_spectral

        return img_raw

    def forward(self, img_nbit, iso):
        """Simulate sensor output with noise and ISP.

        Args:
            img_nbit: Tensor of shape (B, 3, H, W), range [~black_level, 2**bit - 1]
            iso: ISO value as int

        Returns:
            img_noise: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        img_raw = self.response_curve(img_nbit)
        img_noise = self.simu_noise(img_raw, iso)
        img_noise = self.isp(img_noise)
        return img_noise

    # ===============================
    # Unprocess
    # ===============================
    def unprocess(self, image, in_type="rgb"):
        """Unprocess an image to unbalanced RAW RGB space.

        Args:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
            in_type: Input image type, either "rgb" or "linear_rgb"

        Returns:
            image: Tensor of shape (B, 3, H, W), range [0, 1] in raw space
        """
        isp = self.isp

        # Inverse gamma correction
        if in_type == "linear_rgb":
            pass
        elif in_type == "rgb":
            image = isp.gamma.reverse(image)
        else:
            raise ValueError(f"Invalid input type: {in_type}")

        # Inverse color correction matrix
        image = isp.ccm.reverse(image)

        # Inverse auto white balance
        image = isp.awb.reverse(image)  # (B, 3, H, W), [0, 1]

        return image

    def linrgb2bayer(self, img_linrgb):
        """Unprocess the linear RGB image from [0, 1] to [~black_level, 2**bit - 1].

        Args:
            img_linrgb: Tensor of shape (B, 3, H, W), range [0, 1]

        Returns:
            bayer_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]
        """
        black_level = self.black_level
        bit = self.bit

        bayer_float = self.isp.demosaic.reverse(img_linrgb)
        bayer_nbit = bayer_float * (2**bit - 1 - black_level) + black_level
        bayer_nbit = torch.round(bayer_nbit)
        return bayer_nbit

    def sample_augmentation(self):
        """Randomly sample a set of augmentation parameters for ISP modules. Used for data augmentation during training."""
        self.isp.gamma.sample_augmentation()
        self.isp.ccm.sample_augmentation()
        self.isp.awb.sample_augmentation()

    def reset_augmentation(self):
        """Reset parameters for ISP modules. Used for evaluation."""
        self.isp.gamma.reset_augmentation()
        self.isp.ccm.reset_augmentation()
        self.isp.awb.reset_augmentation()

    # ===============================
    # Packing and unpacking
    # ===============================
    def process2rgb(self, image, in_type="rggb"):
        """Process an image to a RGB image.

        Args:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
            in_type: Input image type, either "rggb" or "bayer"

        Returns:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        # Process to RGB
        if in_type == "rggb":
            image = self.isp(self.rggb2bayer(image))
        elif in_type == "bayer":
            image = self.isp(image)
        else:
            raise ValueError(f"Invalid input type: {in_type}")

        return image

    def bayer2rggb(self, bayer_nbit):
        """Convert RAW bayer image to RAW RGGB image.

        Args:
            bayer_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]

        Returns:
            rggb: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        black_level = self.black_level
        bit = self.bit

        if len(bayer_nbit.shape) == 2:
            bayer_nbit = bayer_nbit.unsqueeze(0).unsqueeze(0)
            single_image = True
        else:
            single_image = False

        B, _, H, W = bayer_nbit.shape
        bayer_rggb = torch.zeros(
            (B, 4, H // 2, W // 2), dtype=bayer_nbit.dtype, device=bayer_nbit.device
        )

        bayer_rggb[:, 0, :, :] = bayer_nbit[:, 0, 0:H:2, 0:W:2]
        bayer_rggb[:, 1, :, :] = bayer_nbit[:, 0, 0:H:2, 1:W:2]
        bayer_rggb[:, 2, :, :] = bayer_nbit[:, 0, 1:H:2, 0:W:2]
        bayer_rggb[:, 3, :, :] = bayer_nbit[:, 0, 1:H:2, 1:W:2]

        # Data range [black_level, 2**bit - 1] -> [0, 1]
        rggb = (bayer_rggb - black_level) / (2**bit - 1 - black_level)

        if single_image:
            rggb = rggb.squeeze(0)

        return rggb

    def rggb2bayer(self, rggb):
        """Convert RGGB image to RAW Bayer.

        Args:
            rggb: Tensor of shape [4, H/2, W/2] or [B, 4, H/2, W/2], range [0, 1]

        Returns:
            bayer: Tensor of shape [1, H, W] or [B, 1, H, W], range [~black_level, 2**bit - 1]
        """
        black_level = self.black_level
        bit = self.bit

        if len(rggb.shape) == 3:
            rggb = rggb.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        B, _, H, W = rggb.shape
        bayer = torch.zeros((B, 1, H * 2, W * 2), dtype=rggb.dtype).to(rggb.device)

        bayer[:, 0, 0 : 2 * H : 2, 0 : 2 * W : 2] = rggb[:, 0, :, :]
        bayer[:, 0, 0 : 2 * H : 2, 1 : 2 * W : 2] = rggb[:, 1, :, :]
        bayer[:, 0, 1 : 2 * H : 2, 0 : 2 * W : 2] = rggb[:, 2, :, :]
        bayer[:, 0, 1 : 2 * H : 2, 1 : 2 * W : 2] = rggb[:, 3, :, :]

        # Data range [0, 1] -> [0, 2**bit-1]
        # bayer = torch.round(bayer * (2**bit - 1 - black_level) + black_level)
        bayer = bayer * (2**bit - 1 - black_level) + black_level

        if single_image:
            bayer = bayer.squeeze(0)

        return bayer

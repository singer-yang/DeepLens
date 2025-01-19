"""Differentiable inverse Image Signal Processing (ISP) pipeline."""

import random

import torch


class Inv_ISP:
    """Inverse Image Signal Processing (ISP) pipeline to generate RAW images from RGB images."""

    def __init__(self, bit=10, black_level=64):
        self.bit = bit
        self.black_level = black_level

        self.awb_gains = [2.0, 1.0, 1.8]

        self.ccm_matrix = torch.tensor(
            [
                [1.8506, -0.7920, -0.0605],
                [-0.1562, 1.6455, -0.4912],
                [0.0176, -0.5439, 1.5254],
                [0.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
        )

        self.gamma_param = 2.2

    def __call__(self, *args, **kwds):
        return self.inv_diff_isp(*args, **kwds)

    def inv_diff_isp(
        self,
        rgb,
        augmentation=False,
        inv_gamma=True,
        inv_tone_mapping=False,
        inv_ccm=True,
        inv_awb=True,
    ):
        """Inverse a simple ISP pipeline to convert rgb images to bayer images.

        Reference:
            [1] https://github.com/google-research/google-research/blob/master/unprocessing/process.py


        Args:
            rgb (tensor): RGB images, shape [B, 3, H, W], data range [0, 1], 8-bit.
            inv_tone_mapping (bool): Whether to apply inverse tone mapping.
            inv_gamma (bool): Whether to apply inverse gamma correction.
            inv_ccm (bool): Whether to apply inverse color correction matrix.
            inv_awb (bool): Whether to apply inverse auto white balance.

        Returns:
            bayer (tensor): Bayer images, shape [B, 1, H, W], data range [64, 2^N-1].
        """
        if inv_gamma:
            # Inverse gamma correction
            rgb = self.inv_gamma(rgb, augmentation=augmentation)

        if inv_ccm:
            # Inverse color correction matrix
            rgb = self.inv_ccm(
                rgb, ccm_matrix=self.ccm_matrix, augmentation=augmentation
            )

        if inv_awb:
            # Inverse white balance
            rgb = self.inv_awb(rgb, awb_gains=self.awb_gains, augmentation=augmentation)

        # Inverse demosaic (Re-mosaic)
        bayer = self.inv_demosaic(rgb)

        # Inverse black level compensation
        bayer_Nbit = self.inv_blc(bayer, bit=self.bit, black_level=self.black_level)

        return bayer_Nbit

    def inv_isp_linearRGB(
        self,
        rgb,
        augmentation=False,
        inv_gamma=True,
        inv_ccm=True,
        inv_awb=True,
    ):
        """Simplified inverse ISP pipeline to convert rgb images to linear rgb images.

        Args:
            rgb (tensor): RGB image, shape [B, 3, H, W], data range [0, 1].
            inv_gamma (bool): Whether to apply inverse gamma correction.
            inv_ccm (bool): Whether to apply inverse color correction matrix.
            inv_awb (bool): Whether to apply inverse auto white balance.

        Returns:
            rgb (tensor): Linear RGB image, shape [B, 3, H, W], data range [0, 1].
        """
        if inv_gamma:
            # Inverse gamma correction
            rgb = self.inv_gamma(rgb, augmentation=augmentation)

        if inv_ccm:
            # Inverse color correction matrix
            rgb = self.inv_ccm(
                rgb, ccm_matrix=self.ccm_matrix, augmentation=augmentation
            )

        if inv_awb:
            # Inverse white balance
            rgb = self.inv_awb(rgb, awb_gains=self.awb_gains, augmentation=augmentation)

        # Inverse black level subtraction
        # Do not quantize the image because the goal for unprocessing is to get the raw image
        # rgb_Nbit = self.inv_blc(rgb, bit=self.bit, black_level=self.black_level, quantize=False)

        return rgb

    # ==================================
    # Inverse ISP functions
    # ==================================
    @staticmethod
    def inv_smooth(image, augmentation=False):
        """Approximately inverts a global tone mapping curve y = x^2*(3-2x)

        Ref: https://github.com/google-research/google-research/blob/master/unprocessing/unprocess.py#L72

        Args:
            image (tensor): Image, shape [H, W] or [B, H, W], data range [0, 1].

        Returns:
            image (tensor): Inverted image, shape [H, W] or [B, H, W], data range [0, 1].
        """
        image = torch.clip(image, 0.0, 1.0)
        image_linear = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
        return image_linear

    @staticmethod
    def inv_gamma(rgb, gamma_param=2.2, augmentation=False):
        """Inverse gamma correction.

        Sometimes we want to reduce gamma value to penalize dark regions.

        Ref: https://github.com/google-research/google-research/blob/master/unprocessing/unprocess.py#L78
        """
        if augmentation:
            gamma_param += random.uniform(-0.4, 0.2)

        rgb = torch.clip(rgb, 1e-8) ** gamma_param
        return rgb

    @staticmethod
    def inv_ccm(rgb_corrected, ccm_matrix, augmentation=False):
        """Inverse Color Correction Matrix (Inverse CCM)

        Args:
            rgb_corrected (torch.Tensor): Corrected RGB tensor of shape [B, 3, H, W].
            ccm_matrix (torch.Tensor, optional): Custom CCM matrix of shape [3, 4]. If None, uses the default matrix.

        Returns:
            torch.Tensor: Original RGB tensor of shape [B, 3, H, W].
        """

        # Extract matrix and bias from CCM
        matrix = ccm_matrix[:3, :].float()  # Shape: (3, 3)
        bias = ccm_matrix[3, :].float().view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)

        # Compute the inverse of the CCM matrix
        inv_matrix = torch.inverse(matrix)  # Shape: (3, 3)

        # Prepare rgb_corrected for matrix multiplication
        rgb_corrected_perm = rgb_corrected.permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Subtract bias
        rgb_minus_bias = rgb_corrected_perm - bias.squeeze()

        # Apply Inverse CCM
        rgb_original = torch.matmul(rgb_minus_bias, inv_matrix.T)  # [B, H, W, 3]
        rgb_original = rgb_original.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Clip the values to ensure they are within the valid range
        rgb_original = torch.clamp(rgb_original, 0.0, 1.0)

        return rgb_original

    @staticmethod
    def inv_awb(rgb, awb_gains=(2.0, 1.0, 1.8), augmentation=False):
        """Inverse auto white balance."""
        kr = awb_gains[0]
        kg = awb_gains[1]
        kb = awb_gains[2]

        if augmentation:
            kr += random.uniform(-0.2, 0.2)
            kb += random.uniform(-0.2, 0.2)

        rgb_unbalanced = torch.zeros_like(rgb)
        if len(rgb.shape) == 3:
            rgb_unbalanced[0, :, :] = rgb[0, :, :] / kr
            rgb_unbalanced[1, :, :] = rgb[1, :, :] / kg
            rgb_unbalanced[2, :, :] = rgb[2, :, :] / kb
        else:
            rgb_unbalanced[:, 0, :, :] = rgb[:, 0, :, :] / kr
            rgb_unbalanced[:, 1, :, :] = rgb[:, 1, :, :] / kg
            rgb_unbalanced[:, 2, :, :] = rgb[:, 2, :, :] / kb

        return rgb_unbalanced

    @staticmethod
    def safe_inv_awb(rgb, rgb_gains=(2.0, 1.0, 1.8), augmentation=False):
        """Inverse auto white balance.

        Ref: https://github.com/google-research/google-research/blob/master/unprocessing/unprocess.py#L92C1-L102C28
        """
        if augmentation:
            kr = random.uniform(1.9, 2.4)
            kg = 1.0
            kb = random.uniform(1.5, 1.9)
        else:
            kr = rgb_gains[0]
            kg = rgb_gains[1]
            kb = rgb_gains[2]

        rgb_unbalanced = torch.zeros_like(rgb)
        if len(rgb.shape) == 3:
            gains = (
                torch.tensor([1.0 / kr, 1.0 / kg, 1.0 / kb], device=rgb.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            gray = torch.mean(rgb, dim=0, keepdim=True)
            inflection = 0.9
            mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)

            rgb_unbalanced = rgb * safe_gains
        elif len(rgb.shape) == 4:
            gains = (
                torch.tensor([1.0 / kr, 1.0 / kg, 1.0 / kb], device=rgb.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
            )

            gray = torch.mean(rgb, dim=1, keepdim=True)
            inflection = 0.9
            mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)

            rgb_unbalanced = rgb * safe_gains
        else:
            raise ValueError("Invalid rgb shape")

        return rgb_unbalanced

    @staticmethod
    def inv_demosaic(raw_rgb):
        """Inverse demosaic from RAW RGB to RAW Bayer.

        Args:
            raw_rgb (torch.Tensor): RAW RGB image, shape [3, H, W] or [B, 3, H, W], data range [0, 1].

        Returns:
            torch.Tensor: Bayer image, shape [1, H, W] or [B, 1, H, W], data range [0, 1].
        """
        if raw_rgb.ndim == 3:
            # Input shape: [3, H, W]
            batch_dim = False
            channels, H, W = raw_rgb.shape
        elif raw_rgb.ndim == 4:
            # Input shape: [B, 3, H, W]
            batch_dim = True
            B, channels, H, W = raw_rgb.shape
        else:
            raise ValueError(
                "raw_rgb must have 3 or 4 dimensions corresponding to [3, H, W] or [B, 3, H, W]."
            )

        if channels != 3:
            raise ValueError("raw_rgb must have 3 channels corresponding to RGB.")

        if batch_dim:
            bayer = torch.zeros(
                (B, 1, H, W), dtype=raw_rgb.dtype, device=raw_rgb.device
            )
            bayer[:, 0, 0::2, 0::2] = raw_rgb[:, 0, 0::2, 0::2]
            bayer[:, 0, 0::2, 1::2] = raw_rgb[:, 1, 0::2, 1::2]
            bayer[:, 0, 1::2, 0::2] = raw_rgb[:, 1, 1::2, 0::2]
            bayer[:, 0, 1::2, 1::2] = raw_rgb[:, 2, 1::2, 1::2]
        else:
            bayer = torch.zeros((1, H, W), dtype=raw_rgb.dtype, device=raw_rgb.device)
            bayer[0, 0::2, 0::2] = raw_rgb[0, 0::2, 0::2]
            bayer[0, 0::2, 1::2] = raw_rgb[1, 0::2, 1::2]
            bayer[0, 1::2, 0::2] = raw_rgb[1, 1::2, 0::2]
            bayer[0, 1::2, 1::2] = raw_rgb[2, 1::2, 1::2]

        return bayer

    @staticmethod
    def inv_blc(bayer, bit=10, black_level=64, quantize=True):
        """Inverse black level compensation.

        Args:
            bayer (tensor): Image, data range [0, 1].
            black_level (int): Black level.
            bit (int): Bit.
            quantize (bool): Whether to quantize the image.

        Returns:
            bayer_Nbit (tensor): Image, data range [0, 2**bit-1].
        """
        max_value = 2**bit - 1
        bayer_Nbit = bayer * (max_value - black_level) + black_level
        if quantize:
            bayer_Nbit = torch.round(bayer_Nbit)
        return bayer_Nbit

"""Image Signal Processing (ISP) pipeline converts RAW bayer images to RGB images.

Reference:
    [1] Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2.
"""

import torch
import torch.nn as nn

from .isp_modules import (
    AntiAliasingFilter,
    AutoWhiteBalance,
    BlackLevelCompensation,
    ColorCorrectionMatrix,
    DeadPixelCorrection,
    Demosaic,
    Denoise,
    GammaCorrection,
    LensShadingCorrection,
)


class SimpleISP(nn.Module):
    """Simple ISP pipeline with the most basic modules."""

    def __init__(self, bit=10, black_level=64, bayer_pattern="rggb"):
        super().__init__()

        self.bit = bit
        self.black_level = black_level
        self.bayer_pattern = bayer_pattern

        self.isp = nn.Sequential(
            BlackLevelCompensation(bit=bit, black_level=black_level),
            Demosaic(bayer_pattern=bayer_pattern, method="bilinear"),
            AutoWhiteBalance(awb_method="gray_world"),
            ColorCorrectionMatrix(ccm_matrix=None),
            GammaCorrection(gamma_param=2.2),
        )

    def forward(self, bayer_nbit):
        """Simulate sensor output.

        Args:
            bayer_nbit: Input bayer pattern tensor.

        Returns:
            Processed RGB image.
        """
        return self.isp(bayer_nbit)


class InvertibleISP(nn.Module):
    """Invertible and differentiable ISP pipeline.

    Rerference:
        [1] Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2. (page 23, 50)
    """

    def __init__(self, bit=10, black_level=64, bayer_pattern="rggb"):
        super().__init__()

        self.bit = bit
        self.black_level = black_level
        self.bayer_pattern = bayer_pattern

        self.blc = BlackLevelCompensation(bit=bit, black_level=black_level)
        self.demosaic = Demosaic(bayer_pattern=bayer_pattern, method="3x3")
        self.awb = AutoWhiteBalance(awb_method="manual", gains=(2.0, 1.0, 1.8))
        self.ccm = ColorCorrectionMatrix(ccm_matrix=None)
        self.gamma = GammaCorrection(gamma_param=2.2)

        self.isp = nn.Sequential(
            self.blc,
            self.demosaic,
            self.awb,
            self.ccm,
            self.gamma,
        )

    def forward(self, bayer_nbit):
        """A basic differentiable and invertible ISP pipeline.

        Args:
            bayer_Nbit: Input tensor of shape [B, 1, H, W], data range [~black_level, 2^bit-1].

        Returns:
            rgb: Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        img = self.isp(bayer_nbit)
        return img

    def reverse(self, img):
        """Inverse ISP.

        Args:
            img: Input tensor of shape [B, 3, H, W], data range [0, 1].

        Returns:
            bayer_Nbit: Output tensor of shape [B, 1, H, W], data range [~black_level, 2^bit-1].
        """
        img = self.gamma.reverse(img)
        img = self.ccm.reverse(img)
        img = self.awb.reverse(img)
        bayer = self.demosaic.reverse(img)
        bayer = self.blc.reverse(bayer)
        return bayer


class OpenISP(nn.Module):
    """Image Signal Processing (ISP).

    Reference:
        [1] Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2.
        [2] https://github.com/QiuJueqin/fast-openISP/tree/master
    """

    def __init__(self, bit=10, black_level=64, bayer_pattern="rggb"):
        self.bit = bit
        self.black_level = black_level
        self.bayer_pattern = bayer_pattern

        # DPC
        self.dead_pixel_threshold = 30

        # AAF
        self.raw_denoise_method = "none"  # "bilateral"

        # CFA
        self.bayer_pattern = bayer_pattern
        self.demosaic_method = "bilinear"  # "malvar"

        # AWB
        self.awb_method = "naive"
        self.awb_gains = [2.0, 1.0, 1.8]

        # CCM
        # Reference data from https://github.com/QiuJueqin/fast-openISP/blob/master/configs/nikon_d3200.yaml#L57
        # Alternative data from https://github.com/timothybrooks/hdr-plus/blob/master/src/finish.cpp#L626
        self.ccm_matrix = torch.tensor(
            [
                [1.8506, -0.7920, -0.0605],
                [-0.1562, 1.6455, -0.4912],
                [0.0176, -0.5439, 1.5254],
                [0.0, 0.0, 0.0],
            ]
        )

        # GC
        self.gamma_param = 2.2

        self.isp_pipeline = nn.Sequential(
            # 1. Sensor correction
            DeadPixelCorrection(threshold=30),
            BlackLevelCompensation(bit=bit, black_level=black_level),
            LensShadingCorrection(shading_map=None),
            # 2. Before demosaic, remove moiré pattern, denoise, and deblur
            AntiAliasingFilter(method=None),
            Denoise(method=None),
            # 3. Demosaic, process in rgb space
            Demosaic(bayer_pattern=bayer_pattern, method="bilinear"),
            AutoWhiteBalance(awb_method="gray_world"),
            ColorCorrectionMatrix(ccm_matrix=self.ccm_matrix),
            GammaCorrection(gamma_param=self.gamma_param),
            # 4. Convert to ycrcb space and do image enhancement
        )

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, bayer_nbit):
        """Process single RAW bayer image with a naive ISP pipeline.

        Args:
            bayer_nbit: Input tensor of shape [B, 1, H, W], data range [~black_level, 2^bit-1]

        Returns:
            rgb: RGB image, shape (B, 3, H, W), data range [0, 1].

        Reference:
            [1] https://github.com/QiuJueqin/fast-openISP/tree/master
        """
        return self.isp_pipeline(bayer_nbit)

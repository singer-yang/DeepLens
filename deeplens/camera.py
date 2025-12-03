# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Camera contains a lens and a sensor, working as an image simulator in an end-to-end computational imaging pipeline."""

import torch

from deeplens.sensor import RGBSensor


# ===========================================
# Renderer
# ===========================================
class Renderer:
    """Base class for image simulation and rendering.

    Supports two types of renderers:
        1. Camera renderer using optical simulation.
        2. PSF renderer using calibrated PSF data.
    """

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __call__(self, *args, **kwargs):
        """Alias for ``render()``."""
        return self.render(*args, **kwargs)

    def set_device(self, device):
        """Set the compute device."""
        self.device = device

    def move_to_device(self, data_dict):
        """Move all tensors in the dict to the configured device."""
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device)
        return data_dict

    def render(self, data_dict):
        """Subclasses must implement rendering."""
        raise NotImplementedError


# ===========================================
# Camera renderer
# ===========================================
class Camera(Renderer):
    """Camera system consisting of an optical lens and a sensor.

    This class simulates real camera-captured images for computational imaging
    applications, including lens aberrations and sensor noise characteristics.
    """

    def __init__(
        self,
        lens_file,
        sensor_file,
        lens_type="geolens",
        device=None,
    ):
        super().__init__(device=device)

        # Sensor
        self.sensor = RGBSensor(sensor_file)
        self.sensor.to(device)
        sensor_res = self.sensor.res
        sensor_size = self.sensor.size

        # Lens (here we can use either GeoLens or other lens models)
        if lens_type == "geolens":
            from deeplens.geolens import GeoLens

            self.lens = GeoLens(lens_file, device=device)
        elif lens_type == "hybridlens":
            from deeplens.hybridlens import HybridLens

            self.lens = HybridLens(lens_file, device=device)
        else:
            raise NotImplementedError(f"Unsupported lens type: {lens_type}")
        self.lens.set_sensor(sensor_res=sensor_res, sensor_size=sensor_size)

    def __call__(self, data_dict):
        """Alias for ``render()``."""
        return self.render(data_dict)

    def render(self, data_dict, render_mode="psf_patch", output_type="rggbif"):
        """Simulate camera-captured images with lens aberrations and sensor noise.

        This method performs the complete imaging pipeline: converts input to linear RGB,
        applies lens aberrations, converts to Bayer format, adds sensor noise, and prepares
        output for network training or testing.

        Args:
            data_dict (dict): Dictionary containing essential imaging parameters:
                - "img": sRGB image (torch.Tensor), shape (B, 3, H, W), range [0, 1]
                - "iso": ISO value (int), shape (B,)
                - "field_center": Field center coordinates (torch.Tensor), shape (B, 2), range [-1, 1]
            render_mode (str): Rendering method. Defaults to "psf_patch".
            output_type (str): Output format type. Defaults to "rggbif".

        Returns:
            tuple: (data_lq, data_gt)
                - data_lq: Low-quality network input with degradations
                - data_gt: Ground-truth data for training

        References:
            [1] "Unprocessing Images for Learned Raw Denoising", CVPR 2018.
            [2] "Optical Aberration Correction in Postprocessing using Imaging Simulation", SIGGRAPH 2021.
            [3] "Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur", ICCV Workshop 2025.
        """
        data_dict = self.move_to_device(data_dict)
        img = data_dict["img"]
        iso = data_dict["iso"]

        # Unprocess from RGB to linear RGB space
        sensor = self.sensor
        img_linrgb = sensor.unprocess(img)  # (B, 3, H, W), [0, 1]

        # Lens aberration simulation in linear RGB space
        img_lq = self.render_lens(
            img_linrgb, render_mode=render_mode, **data_dict
        )  # (B, 3, H, W), [0, 1]

        # Convert linear RGB to Bayer space
        bayer_gt = sensor.linrgb2bayer(img_linrgb)  # (B, 1, H, W), [0, 2**bit - 1]
        bayer_lq = sensor.linrgb2bayer(img_lq)  # (B, 1, H, W), [0, 2**bit - 1]

        # Simulate sensor noise
        bayer_lq = sensor.simu_noise(
            bayer_lq, iso
        )  # (B, 1, H, W), [black_level, 2**bit - 1]

        # Pack output for network training
        data_lq, data_gt = self.pack_output(
            bayer_gt=bayer_gt,
            bayer_lq=bayer_lq,
            output_type=output_type,
            **data_dict,
        )
        return data_lq, data_gt

    def render_lens(self, img_linrgb, render_mode="psf_patch", **kwargs):
        """Apply lens aberrations to a linear RGB image.

        Args:
            img_linrgb (torch.Tensor): Linear RGB image (energy representation),
                shape (B, 3, H, W), range [0, 1]
            render_mode (str): Rendering method to use. Options include:
                - "psf_patch": PSF with patch rendering
                - "psf_map": PSF map rendering
                - "psf_pixel": Pixel-wise PSF rendering
                - "ray_tracing": Full ray tracing simulation
                - "psf_patch_depth_interp": PSF patch with depth interpolation
                Defaults to "psf_patch".
            **kwargs: Additional method-specific arguments (e.g., field_center, depth).

        Returns:
            torch.Tensor: Degraded image with lens aberrations, shape (B, 3, H, W), range [0, 1]
        """
        if render_mode == "psf_patch":
            # Because different image in a batch can have different PSF, so we use for loop here
            img_lq_ls = []
            for b in range(img_linrgb.shape[0]):
                img = img_linrgb[b, ...].unsqueeze(0)
                psf_center = kwargs["field_center"][b, ...]
                img_lq = self.lens.render(
                    img, method="psf_patch", psf_center=psf_center
                )
                img_lq_ls.append(img_lq)
            img_lq = torch.cat(img_lq_ls, dim=0)

        elif render_mode == "psf_map":
            img_lq = self.lens.render(img_linrgb, method="psf_map")

        elif render_mode == "psf_pixel":
            depth = kwargs["depth"][b, ...]
            img_lq = self.lens.render(img_linrgb, method="psf_pixel", **kwargs)

        elif render_mode == "ray_tracing":
            img_lq = self.lens.render(img_linrgb, method="ray_tracing", **kwargs)

        elif render_mode == "psf_patch_depth_interp":
            img_lq_ls = []
            for b in range(img_linrgb.shape[0]):
                img = img_linrgb[b, ...].unsqueeze(0)
                psf_center = kwargs["field_center"][b, ...].unsqueeze(0)
                depth = kwargs["depth"][b, ...].unsqueeze(0)
                img_lq = self.lens.render_rgbd(
                    img, depth, method="psf_patch", psf_center=psf_center
                )
                img_lq_ls.append(img_lq)
            img_lq = torch.cat(img_lq_ls, dim=0)

        else:
            raise NotImplementedError(f"Invalid render mode: {render_mode}")

        return img_lq

    def pack_output(
        self,
        bayer_lq,
        bayer_gt,
        iso,
        iso_scale=1000,
        output_type="rggbi",
        **kwargs,
    ):
        """Pack Bayer data into network-ready inputs and targets.

        Args:
            bayer_lq (torch.Tensor): Noisy Bayer image, shape (B, 1, H, W), range [~black_level, 2**bit - 1]
            bayer_gt (torch.Tensor): Clean Bayer image, shape (B, 1, H, W), range [~black_level, 2**bit - 1]
            iso (torch.Tensor): ISO values, shape (B,)
            iso_scale (int): Normalization factor for ISO values. Defaults to 1000.
            output_type (str): Output format specification. Options:
                - "rgb": Standard RGB format
                - "rggbi": RGGB channels + ISO channel (5 channels)
                - "rggbif": RGGB channels + ISO + field position (6 channels)
                Defaults to "rggbi".
            **kwargs: Additional data required for specific output types (e.g., field_center).

        Returns:
            *_lq (torch.Tensor): Low-quality network input (B, C, H, W)
            *_gt (torch.Tensor): Ground-truth for training (B, C, H, W)
        """
        sensor = self.sensor
        pixel_size = sensor.pixel_size
        device = bayer_lq.device

        # Prepare network input
        if output_type == "rgb":
            rgb_gt = sensor.isp(bayer_gt)
            rgb_lq = sensor.isp(bayer_lq)
            return rgb_lq, rgb_gt

        elif output_type == "rggbi":
            # RGGB channels
            rggb_gt = sensor.bayer2rggb(bayer_gt)  # (B, 4, H, W), [0, 1]
            rggb_lq = sensor.bayer2rggb(bayer_lq)  # (B, 4, H, W), [0, 1]
            B, _, H, W = rggb_lq.shape

            # ISO channel (B, 1, H, W)
            iso_channel = (iso / iso_scale).view(-1, 1, 1, 1).repeat(1, 1, H, W)

            # Concatenate to RGGBI 5 channels
            rggbi_lq = torch.cat([rggb_lq, iso_channel], dim=1)
            return rggbi_lq, rggb_gt

        elif output_type == "rggbif":
            # RGGB channels
            rggb_gt = sensor.bayer2rggb(bayer_gt)  # (B, 4, H, W), [0, 1]
            rggb_lq = sensor.bayer2rggb(bayer_lq)  # (B, 4, H, W), [0, 1]
            B, _, H, W = rggb_lq.shape

            # ISO channel (B, 1, H, W)
            iso_channel = (iso / iso_scale).view(-1, 1, 1, 1).repeat(1, 1, H, W)

            # Field channel (B, 1, H, W)
            field_channels = []
            for b in range(B):
                field_center = kwargs["field_center"][b, ...]
                # After shuffling to rggb, the stride (step size) is 2 * pixel_size
                grid_x, grid_y = torch.meshgrid(
                    torch.linspace(
                        field_center[0] - W * pixel_size,
                        field_center[0] + W * pixel_size,
                        W,
                        device=device,
                    ),
                    torch.linspace(
                        field_center[1] + H * pixel_size,
                        field_center[1] - H * pixel_size,
                        H,
                        device=device,
                    ),
                    indexing="xy",
                )
                field_channel = torch.sqrt(grid_x**2 + grid_y**2).unsqueeze(0)
                field_channels.append(field_channel)
            field_channel = torch.cat(field_channels, dim=0).unsqueeze(1)

            # Concatenate to RGGBIF 6 channels
            rggbif_lq = torch.cat([rggb_lq, iso_channel, field_channel], dim=1)
            return rggbif_lq, rggb_gt

        else:
            raise NotImplementedError(f"Invalid output type: {output_type}")

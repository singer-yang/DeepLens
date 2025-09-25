# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Camera contains a lens and a sensor, working as an image simulator in an end-to-end computational imaging pipeline.
"""

import torch

from deeplens.geolens import GeoLens
from deeplens.sensor import RGBSensor


# ===========================================
# Renderer
# ===========================================
class Renderer:
    """Renderer is the basic class for image simulation. 
    
    We will support two types of renderers:
        [1] Camera renderer using optical simulation.
        [2] PSF renderer using calibrated PSF data.
    """
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __call__(self, *args, **kwargs):
        """Render a blurry and noisy RGB image batch with for loop."""
        return self.render(*args, **kwargs)

    def set_device(self, device):
        """Set the device for rendering."""
        self.device = device

    def move_to_device(self, data_dict):
        """Move data to device."""
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device)
        return data_dict

    def render_single_frame(self, *args, **kwargs):
        """Render a single frame of a blurry and noisy RGB image from spectral data."""
        raise NotImplementedError
    
    def render(self, data_dict):
        """Render a blurry and noisy RGB image batch with for loop."""
        raise NotImplementedError


# ===========================================
# Camera renderer
# ===========================================
class Camera(Renderer):
    """Camera includes an optical lens and a sensor. It is used to simulate real camera-captured images for computational imaging."""
    def __init__(
        self,
        lens_file,
        sensor_size=(8.0, 6.0),
        sensor_res=(4032, 3024),
        device=None,
    ):
        super().__init__(device=device)

        # Lens (here we can use either GeoLens or other lens models)
        self.lens = GeoLens(lens_file, device=device)
        self.lens.set_sensor(sensor_res=sensor_res, sensor_size=sensor_size)

        # Sensor
        self.sensor = RGBSensor(
            res=sensor_res,
            bit=10,
            black_level=64,
            iso_base=100,
            read_noise_std=0.5,
            shot_noise_std_alpha=0.4,
        ).to(device)

    def __call__(self, data_dict):
        """Simulate a blurry and noisy RGB image considering lens and sensor."""
        return self.render(data_dict)

    def render(self, data_dict, render_mode="psf_patch", output_type="rggbif"):
    # def render(self, data_dict):
        """Simulate image with lens aberrations and sensor noise in RAW space. 

        Args:
            data_dict (dict): Dictionary containing essential information for image simulation. 
                For example:
                    {
                        "img": rgb image (torch.Tensor), (B, 3, H, W), [0, 1]
                        "iso": iso (int), (B,)
                        "field_center": field_center (torch.Tensor), (B, 2), [-1, 1]
                    }

        Returns:
            img_rgb (torch.Tensor): RGB image (B, 3, H, W), [0, 1]

        Reference:
            [1] "Unprocessing Images for Learned Raw Denoising", CVPR 2018.
            [2] "Optical Aberration Correction in Postprocessing using Imaging Simulation", SIGGRAPH 2021.
            [3] "Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur", ICCV Workshop 2025.
        """
        data_dict = self.move_to_device(data_dict)
        img = data_dict["img"]
        iso = data_dict["iso"]

        # Unprocess from RGB to RAW (linear RGB) space
        sensor = self.sensor
        img_raw = sensor.unprocess(img)  # (B, 3, H, W), [0, 1]

        # Lens aberration simulation in RAW (linear RGB) space
        img_lq = self.render_lens(img_raw, render_mode=render_mode, **data_dict)  # (B, 3, H, W), [0, 1]

        # Convert to Bayer space
        bayer_gt = sensor.raw2bayer(img_raw)  # (B, 1, H, W), [0, 2**bit - 1]
        bayer_lq = sensor.raw2bayer(img_lq)  # (B, 1, H, W), [0, 2**bit - 1]

        # Simulate sensor noise
        bayer_lq = sensor.simu_noise(bayer_lq, iso)  # (B, 1, H, W), [black_level, 2**bit - 1]

        # Pack output for network training
        data_lq, data_gt = self.pack_output(
            bayer_gt=bayer_gt,
            bayer_lq=bayer_lq,
            output_type=output_type,
            **data_dict,
        )
        return data_lq, data_gt

    def render_lens(self, img_raw, render_mode="psf_patch", **kwargs):
    # def render_lens(self, data_dict):
        """Simulate an image with lens aberrations.

        Args:
            img_raw (torch.Tensor): Raw image (linear RGB, or image after demosaic). (B, 3, H, W), [0, 1]
            render_mode (str): Render mode. Defaults to "psf_patch".
            **kwargs: Additional arguments for different methods.

        Returns:
            img_lq (torch.Tensor): Low-quality image (B, 3, H, W), [0, 1]
        """
        # img = data_dict["img"]
        # render_mode = data_dict["render_mode"]
        # kwargs = data_dict
        
        if render_mode == "psf_patch":
            # Because different image in a batch can have different PSF, so we should use for loop
            img_lq_ls = []
            for b in range(img_raw.shape[0]):
                img = img_raw[b, ...].unsqueeze(0)
                psf_center = kwargs["field_center"][b, ...]
                img_lq = self.lens.render(img, method="psf_patch", psf_center=psf_center)
                img_lq_ls.append(img_lq)
            img_lq = torch.cat(img_lq_ls, dim=0)
            
        elif render_mode == "psf_map":
            img_lq = self.lens.render(img, method="psf_map")
        
        elif render_mode == "psf_pixel":
            depth = kwargs["depth"][b, ...]
            img_lq = self.lens.render(img, method="psf_pixel", **kwargs)
        
        elif render_mode == "ray_tracing":
            img_lq = self.lens.render(img, method="ray_tracing", **kwargs)
        
        elif render_mode == "psf_patch_depth_interp":
            img_lq_ls = []
            for b in range(img_raw.shape[0]):
                img = img_raw[b, ...].unsqueeze(0)
                psf_center = kwargs["field_center"][b, ...]
                depth = kwargs["depth"][b, ...]
                img_lq = self.lens.render_rgbd(img, depth, method="psf_patch", psf_center=psf_center)
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
        """Pack Prepare network input for training and testing.

        Args:
            bayer_lq (torch.Tensor): Bayer image with noise (B, 1, H, W), [~black_level, 2**bit - 1]
            bayer_gt (torch.Tensor): Bayer image (B, 1, H, W), [~black_level, 2**bit - 1]
            iso (torch.Tensor): ISO value (B,)
            iso_scale (int): ISO scale. Defaults to 1000.
            output_type (str): Output type. Defaults to "rggbi".
            **kwargs: Additional arguments.

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
            rggbif_lq = torch.cat(
                [rggb_lq, iso_channel, field_channel], dim=1
            )
            return rggbif_lq, rggb_gt

        else:
            raise NotImplementedError(f"Invalid output type: {output_type}")

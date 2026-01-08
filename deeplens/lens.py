# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Base class for optical lens. When creating a new lens (geolens, diffractivelens, etc.), it should inherit from the Lens class and rewrite core functions."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from deeplens.basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_PSF,
    WAVE_RGB,
    DeepObj,
    init_device,
)
from deeplens.optics.psf import (
    conv_psf,
    conv_psf_depth_interp,
    conv_psf_map,
    conv_psf_map_depth_interp,
    conv_psf_pixel,
)


class Lens(DeepObj):
    def __init__(self, dtype=torch.float32, device=None):
        """Initialize a lens class.

        Args:
            dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
            device (str, optional): Device to run the lens. Defaults to None.
        """
        # Lens device
        if device is None:
            self.device = init_device()
        else:
            self.device = torch.device(device)

        # Lens default dtype
        self.dtype = dtype

    def read_lens_json(self, filename):
        """Read lens from a json file."""
        raise NotImplementedError

    def write_lens_json(self, filename):
        """Write lens to a json file."""
        raise NotImplementedError

    def set_sensor(self, sensor_size, sensor_res):
        """Set sensor size and resolution.

        Args:
            sensor_size (tuple): Sensor size (w, h) in [mm].
            sensor_res (tuple): Sensor resolution (W, H) in [pixels].
        """
        assert sensor_size[0] * sensor_res[1] == sensor_size[1] * sensor_res[0], (
            "Sensor resolution aspect ratio does not match sensor size aspect ratio."
        )
        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]
        self.r_sensor = float(np.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2)) / 2
        self.calc_fov()

    def set_sensor_res(self, sensor_res):
        """Set sensor resolution (and aspect ratio) while keeping sensor radius unchanged.

        Args:
            sensor_res (tuple): Sensor resolution (W, H) in [pixels].
        """
        # Change sensor resolution
        self.sensor_res = sensor_res

        # Change sensor size (r_sensor is fixed)
        diam_res = float(np.sqrt(self.sensor_res[0] ** 2 + self.sensor_res[1] ** 2))
        self.sensor_size = (
            2 * self.r_sensor * self.sensor_res[0] / diam_res,
            2 * self.r_sensor * self.sensor_res[1] / diam_res,
        )
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]
        self.calc_fov()

    @torch.no_grad()
    def calc_fov(self):
        """Compute FoV (radian) of the lens.

        Reference:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        self.vfov = 2 * float(np.atan(self.sensor_size[0] / 2 / self.foclen))
        self.hfov = 2 * float(np.atan(self.sensor_size[1] / 2 / self.foclen))
        self.dfov = 2 * float(np.atan(self.r_sensor / self.foclen))
        self.rfov = self.dfov / 2  # radius (half diagonal) FoV

    # ===========================================
    # PSF-ralated functions
    # 1. Point PSF
    # 2. PSF map
    # 3. PSF radial
    # ===========================================
    def psf(self, points, wvln=DEFAULT_WAVE, ks=PSF_KS, **kwargs):
        """Compute monochrome point PSF.

        NOTE:
            [1] This function should be designed to be differentiable.
            [2] For each point source, we should consider diffraction in the calculation, even for incoherent imaging.

        Args:
            points (tensor): Shape of [N, 3] or [3].
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.
            ks (int, optional): Kernel size. Defaults to PSF_KS.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].

        Reference:
            [1] Cittert-Zernike Theorem.
        """
        raise NotImplementedError

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """Compute RGB point PSF.

        Args:
            points (tensor): Shape of [N, 3] or [3].
            ks (int, optional): Kernel size. Defaults to 51.

        Returns:
            psf_rgb: Shape of [N, 3, ks, ks] or [3, ks, ks].
        """
        psfs = []
        for wvln in WAVE_RGB:
            psfs.append(self.psf(points=points, ks=ks, wvln=wvln, **kwargs))
        psf_rgb = torch.stack(psfs, dim=-3)  # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf_rgb

    def point_source_grid(
        self, depth, grid=(9, 9), normalized=True, quater=False, center=True
    ):
        """Generate point source grid for PSF calculation.

        Args:
            depth (float): Depth of the point source.
            grid (tuple): Grid size (grid_w, grid_h). Defaults to (9, 9), meaning 9x9 grid.
            normalized (bool): Return normalized object source coordinates. Defaults to True, meaning object sources xy coordinates range from [-1, 1].
            quater (bool): Use quater of the sensor plane to save memory. Defaults to False.
            center (bool): Use center of each patch. Defaults to True.

        Returns:
            point_source: Normalized object source coordinates. Shape of [grid_h, grid_w, 3], [-1, 1], [-1, 1], [-Inf, 0].
        """
        # Compute point source grid
        if grid[0] == 1:
            x, y = torch.tensor([[0.0]]), torch.tensor([[0.0]])
            assert not quater, "Quater should be False when grid is 1."
        else:
            if center:
                # Use center of each patch
                half_bin_size = 1 / 2 / (grid[0] - 1)
                x, y = torch.meshgrid(
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid[0]),
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid[1]),
                    indexing="xy",
                )
            else:
                # Use corner of image sensor
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid[0]),
                    torch.linspace(0.98, -0.98, grid[1]),
                    indexing="xy",
                )

        z = torch.full_like(x, depth)
        point_source = torch.stack([x, y, z], dim=-1)

        # Use quater of the sensor plane to save memory
        if quater:
            z = torch.full_like(x, depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid[0] // 2 if grid[0] % 2 == 0 else grid[0] // 2 + 1
            bound_j = grid[1] // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        # De-normalize object source coordinates to physical coordinates
        if not normalized:
            scale = self.calc_scale(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    def psf_map(self, grid=(5, 5), wvln=DEFAULT_WAVE, depth=DEPTH, ks=PSF_KS, **kwargs):
        """Compute monochrome PSF map.

        Args:
            grid (tuple): Grid size (grid_w, grid_h). Defaults to (5, 5), meaning 5x5 grid.
            wvln (float): Wavelength. Defaults to DEFAULT_WAVE.
            depth (float): Depth of the object. Defaults to DEPTH.
            ks (int): Kernel size. Defaults to PSF_KS.

        Returns:
            psf_map: Shape of [grid_h, grid_w, 3, ks, ks].
        """
        # PSF map grid
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3)

        # Compute PSF map
        psfs = []
        for i in range(points.shape[0]):
            point = points[i, ...]
            psf = self.psf(points=point, wvln=wvln, ks=ks)
            psfs.append(psf)
        psf_map = torch.stack(psfs).unsqueeze(1)  # shape [grid_h * grid_w, 1, ks, ks]

        # Reshape PSF map from [grid_h * grid_w, 1, ks, ks] -> [grid_h, grid_w, 1, ks, ks]
        psf_map = psf_map.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    def psf_map_rgb(self, grid=(5, 5), ks=PSF_KS, depth=DEPTH, **kwargs):
        """Compute RGB PSF map.

        Args:
            grid (tuple): Grid size (grid_w, grid_h). Defaults to (5, 5), meaning 5x5 grid.
            ks (int): Kernel size. Defaults to 51, meaning 51x51 kernel size.
            depth (float): Depth of the object. Defaults to DEPTH.
            **kwargs: Additional arguments for psf_map().

        Returns:
            psf_map: Shape of [grid_h, grid_w, 3, ks, ks].
        """
        psfs = []
        for wvln in WAVE_RGB:
            psf_map = self.psf_map(grid=grid, ks=ks, depth=depth, wvln=wvln, **kwargs)
            psfs.append(psf_map)
        psf_map = torch.cat(psfs, dim=2)  # shape [grid_h, grid_w, 3, ks, ks]
        return psf_map

    @torch.no_grad()
    def draw_psf_map(
        self,
        grid=(7, 7),
        ks=PSF_KS,
        depth=DEPTH,
        log_scale=False,
        save_name="./psf_map.png",
        show=False,
    ):
        """Draw RGB PSF map of the lens."""
        # Calculate RGB PSF map, shape [grid_h, grid_w, 3, ks, ks]
        psf_map = self.psf_map_rgb(depth=depth, grid=grid, ks=ks)

        # Create a grid visualization (vis_map: shape [3, grid_h * ks, grid_w * ks])
        grid_w, grid_h = grid if isinstance(grid, tuple) else (grid, grid)
        h, w = grid_h * ks, grid_w * ks
        vis_map = torch.zeros((3, h, w), device=psf_map.device, dtype=psf_map.dtype)

        # Put each PSF into the vis_map
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract the PSF at this grid position
                psf = psf_map[i, j]  # shape [3, ks, ks]

                # Normalize the PSF
                if log_scale:
                    # Log scale normalization for better visualization
                    psf = torch.log(psf + 1e-4)  # 1e-4 is an empirical value
                    psf = (psf - psf.min()) / (psf.max() - psf.min() + 1e-8)
                else:
                    # Linear normalization
                    local_max = psf.max()
                    if local_max > 0:
                        psf = psf / local_max

                # Place the normalized PSF in the visualization map
                y_start, y_end = i * ks, (i + 1) * ks
                x_start, x_end = j * ks, (j + 1) * ks
                vis_map[:, y_start:y_end, x_start:x_end] = psf

        # Create the figure and display
        fig, ax = plt.subplots(figsize=(10, 10))

        # Convert to numpy for plotting
        vis_map = vis_map.permute(1, 2, 0).cpu().numpy()
        ax.imshow(vis_map)

        # Add scale bar near bottom-left
        H, W, _ = vis_map.shape
        scale_bar_length = 100
        arrow_length = scale_bar_length / (self.pixel_size * 1e3)
        y_position = H - 20  # a little above the lower edge
        x_start = 20
        x_end = x_start + arrow_length

        ax.annotate(
            "",
            xy=(x_start, y_position),
            xytext=(x_end, y_position),
            arrowprops=dict(arrowstyle="-", color="white"),
            annotation_clip=False,
        )
        ax.text(
            x_end + 5,
            y_position,
            f"{scale_bar_length} μm",
            color="white",
            fontsize=12,
            ha="left",
            va="center",
            clip_on=False,
        )

        # Clean up axes and save
        ax.axis("off")
        plt.tight_layout(pad=0)

        if show:
            return fig, ax
        else:
            plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def point_source_radial(self, depth, grid=9, center=False):
        """Compute point radial [0, 1] in the object space to compute PSF grid.

        Args:
            grid (int, optional): Grid size. Defaults to 9.

        Returns:
            point_source: Shape of [grid, 3].
        """
        if grid == 1:
            x = torch.tensor([0.0])
        else:
            # Select center of bin to calculate PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x = torch.linspace(0, 1 - half_bin_size, grid)
            else:
                x = torch.linspace(0, 0.98, grid)

        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source

    @torch.no_grad()
    def draw_psf_radial(
        self, M=3, depth=DEPTH, ks=PSF_KS, log_scale=False, save_name="./psf_radial.png"
    ):
        """Draw radial PSF (45 deg). Will draw M PSFs, each of size ks x ks."""
        x = torch.linspace(0, 1, M)
        y = torch.linspace(0, 1, M)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)

        psfs = []
        for i in range(M):
            # Scale PSF for a better visualization
            psf = self.psf_rgb(points=points[i], ks=ks, recenter=True, spp=SPP_PSF)
            psf /= psf.max()

            if log_scale:
                psf = torch.log(psf + EPSILON)
                psf = (psf - psf.min()) / (psf.max() - psf.min())

            psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)

    # ===========================================
    # Image simulation-ralated functions
    # ===========================================

    # -------------------------------------------
    # Simulate 2D scene
    # -------------------------------------------
    def render(self, img_obj, depth=DEPTH, method="psf_patch", **kwargs):
        """Differentiable image simulation, considering only 2D scene.

        NOTE:
            [1] This function performs only the optical component of image simulation and is designed to be fully differentiable. Other components (e.g., noise simulation) are handled by other functions (see Camera class).
            [2] For incoherent imaging, we should calculate the intensity PSF (squared magnitude of the complex amplitude) and convolve it with the object-space image. For coherent imaging, we should convolve the complex PSF with the complex object image and then calculate the intensity by squaring the magnitude.

        Image simulation methods:
            [1] PSF map, convolution by patches.
            [2] PSF patch, convolution by a single PSF.
            [3] Ray tracing rendering, in GeoLens.
            [4] ...

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. Defaults to "psf".
            **kwargs: Additional arguments for different methods.

        Returns:
            img_render (tensor): Rendered image. Shape of [N, C, H, W].

        Reference:
            [1] "Optical Aberration Correction in Postprocessing using Imaging Simulation", TOG 2021.
            [2] "Efficient depth- and spatially-varying image simulation for defocus deblur", ICCVW 2025.
        """
        # Check sensor resolution
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # Image simulation (in RAW space)
        if method == "psf_map":
            # Render full resolution image with PSF map convolution
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", 51)
            img_render = self.render_psf_map(
                img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
            )

        elif method == "psf_patch":
            # Render an image patch with its corresponding PSF
            psf_center = kwargs.get("psf_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", 51)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
            )

        elif method == "psf_pixel":
            raise NotImplementedError(
                "Per-pixel PSF convolution has not been implemented."
            )

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_psf(self, img_obj, depth=DEPTH, psf_center=(0, 0), psf_ks=PSF_KS):
        """Render image patch using PSF convolution. Better not use this function to avoid confusion."""
        return self.render_psf_patch(
            img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
        )

    def render_psf_patch(self, img_obj, depth=DEPTH, psf_center=(0, 0), psf_ks=PSF_KS):
        """Render an image patch using PSF convolution, and return positional encoding channel.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [B, C, H, W].
            depth (float): Depth of the object.
            psf_center (tensor): Center of the PSF patch. Shape of [2] or [B, 2].
            psf_ks (int): PSF kernel size. Defaults to PSF_KS.

        Returns:
            img_render: Rendered image. Shape of [B, C, H, W].
        """
        # Convert psf_center to tensor
        if isinstance(psf_center, (list, tuple)):
            points = (psf_center[0], psf_center[1], depth)
            points = torch.tensor(points).unsqueeze(0)
        elif isinstance(psf_center, torch.Tensor):
            depth = torch.full_like(psf_center[..., 0], depth)
            points = torch.stack(
                [psf_center[..., 0], psf_center[..., 1], depth], dim=-1
            )
        else:
            raise Exception(
                f"PSF center must be a list or tuple or tensor, but got {type(psf_center)}."
            )

        # Compute PSF and perform PSF convolution
        psf = self.psf_rgb(points=points, ks=psf_ks).squeeze(0)
        img_render = conv_psf(img_obj, psf=psf)
        return img_render

    def render_psf_map(self, img_obj, depth=DEPTH, psf_grid=7, psf_ks=PSF_KS):
        """Render image using PSF block convolution.

        Note:
            Larger psf_grid and psf_ks are typically better for more accurate rendering, but slower.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [B, C, H, W].
            depth (float): Depth of the object.
            psf_grid (int): PSF grid size.
            psf_ks (int): PSF kernel size. Defaults to PSF_KS.

        Returns:
            img_render: Rendered image. Shape of [B, C, H, W].
        """
        psf_map = self.psf_map_rgb(grid=psf_grid, ks=psf_ks, depth=depth)
        img_render = conv_psf_map(img_obj, psf_map)
        return img_render

    # -------------------------------------------
    # Simulate 3D scene
    # -------------------------------------------
    def render_rgbd(self, img_obj, depth_map, method="psf_patch", **kwargs):
        """Render RGBD image.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [B, C, H, W].
            depth_map (tensor): Depth map. Shape of [B, 1, H, W].
            method (str, optional): Image simulation method. Defaults to "psf_patch".
            **kwargs: Additional arguments for different methods.

        Returns:
            img_render: Rendered image. Shape of [B, C, H, W].

        Reference:
            [1] "Aberration-Aware Depth-from-Focus", TPAMI 2023.
            [2] "Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur", ICCVW 2025.
        """
        depth_map = -1.0 * depth_map

        if method == "psf_patch":
            # Render a small image patch (same FoV, different depth)
            psf_center = kwargs.get("psf_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            depth_min = kwargs.get("depth_min", depth_map.min())
            depth_max = kwargs.get("depth_max", depth_map.max())
            num_depth = kwargs.get("num_depth", 10)

            # Calculate PSF at different depths, (num_depth, 3, ks, ks)
            depths_ref = torch.linspace(depth_min, depth_max, num_depth).to(self.device)
            points = torch.stack(
                [
                    torch.full_like(depths_ref, psf_center[0]),
                    torch.full_like(depths_ref, psf_center[1]),
                    depths_ref,
                ],
                dim=-1,
            )
            psfs = self.psf_rgb(points=points, ks=psf_ks)

            # Image simulation
            img_render = conv_psf_depth_interp(img_obj, depth_map, psfs, depths_ref)
            return img_render

        elif method == "psf_map":
            # Render full resolution image with PSF map convolution
            psf_grid = kwargs.get("psf_grid", (10, 10))  # (grid_w, grid_h)
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            depth_min = kwargs.get("depth_min", depth_map.min())
            depth_max = kwargs.get("depth_max", depth_map.max())
            num_depth = kwargs.get("num_depth", 16)
            depths_ref = torch.linspace(depth_min, depth_max, num_depth).to(self.device)

            # Calculate PSF map at different depths
            psf_maps = []
            for depth in tqdm(depths_ref):
                psf_map = self.psf_map_rgb(grid=psf_grid, ks=psf_ks, depth=depth)
                psf_maps.append(psf_map)
            psf_map = torch.stack(
                psf_maps, dim=2
            )  # shape [grid_h, grid_w, num_depth, 3, ks, ks]

            # Image simulation
            img_render = conv_psf_map_depth_interp(
                img_obj, depth_map, psf_map, depths_ref
            )
            return img_render

        elif method == "psf_pixel":
            # Render full resolution image with pixel-wise PSF convolution. This method is computationally expensive.
            psf_ks = kwargs.get("psf_ks", 32)
            assert img_obj.shape[0] == 1, "Now only support batch size 1"

            # Calculate points in the object space
            points_xy = torch.meshgrid(
                torch.linspace(-1, 1, img_obj.shape[-1], device=self.device),
                torch.linspace(1, -1, img_obj.shape[-2], device=self.device),
                indexing="xy",
            )
            points_xy = torch.stack(points_xy, dim=0).unsqueeze(0)
            points = torch.cat([points_xy, depth_map], dim=1)  # shape [B, 3, H, W]

            # Calculate PSF at different pixels. This step is the most time-consuming.
            points = points.permute(0, 2, 3, 1).reshape(-1, 3)  # shape [H*W, 3]
            psfs = self.psf_rgb(points=points, ks=psf_ks)  # shape [H*W, 3, ks, ks]
            psfs = psfs.reshape(
                img_obj.shape[-2], img_obj.shape[-1], 3, psf_ks, psf_ks
            )  # shape [H, W, 3, ks, ks]

            # Image simulation
            img_render = conv_psf_pixel(img_obj, psfs)  # shape [1, C, H, W]
            return img_render

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

    # ===========================================
    # Optimization-ralated functions
    # ===========================================
    def activate_grad(self, activate=True):
        """Activate gradient for each surface."""
        raise NotImplementedError

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-3]):
        """Get optimizer parameters for different lens parameters."""
        raise NotImplementedError

    def get_optimizer(self, lr=[1e-4, 1e-4, 0, 1e-3]):
        """Get optimizer."""
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)
        return optimizer

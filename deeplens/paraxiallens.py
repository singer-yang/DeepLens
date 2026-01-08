# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Paraxial geometric/ABCD matrix lens model. The paraxial lens model can simulate defocus (Circle of Confusion) but not optical aberrations. This model is commonly used in software such as Blender.

Reference:
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
    [2] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
"""

import numpy as np
import torch

from deeplens.lens import Lens
from deeplens.basics import DEPTH, EPSILON, PSF_KS
from deeplens.optics.psf import conv_psf_depth_interp


class ParaxialLens(Lens):
    def __init__(self, foclen, fnum, sensor_size=None, sensor_res=None, device="cpu"):
        """Initialize a paraxial lens.

        Args:
            foclen (float): Focal length in [mm].
            fnum (float): F-number.
            sensor_size (tuple, optional): Physical sensor size as (W, H) in [mm]. Defaults to (8.0, 8.0).
            sensor_res (tuple, optional): Sensor resolution as (W, H) in pixels. Defaults to (2000, 2000).
            device (str, optional): Computation device. Defaults to "cpu".
        """
        super(ParaxialLens, self).__init__(device=device)

        # Lens parameters
        self.foclen = foclen  # Focal length [mm]
        self.fnum = fnum

        # Sensor size and resolution with defaults
        if sensor_size is None:
            sensor_size = (8.0, 8.0)
            print(
                f"Sensor_size not provided. Using default: {sensor_size} mm. "
                "Use set_sensor() to change."
            )
        if sensor_res is None:
            sensor_res = (2000, 2000)
            print(
                f"Sensor_res not provided. Using default: {sensor_res} pixels. "
                "Use set_sensor() to change."
            )

        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]  # Pixel size [mm]

        self.d_far = -20000.0
        self.d_close = -200.0
        self.refocus(foc_dist=-20000)

    def refocus(self, foc_dist):
        """Refocus the lens to the given focus distance."""
        assert foc_dist < self.foclen, "Focus distance is too close."
        self.foc_dist = foc_dist

    # ===========================================
    # PSF-related functions
    # ===========================================

    def psf(self, points, ks=PSF_KS, psf_type="gaussian", **kwargs):
        """PSF is modeled as a 2D uniform circular disk with diameter CoC.

        Args:
            points (torch.Tensor): Points of the object. Shape [N, 3] or [3].
            ks (int): Kernel size.
            psf_type (str): PSF type. "gaussian" or "pillbox".
            **kwargs: Additional arguments for psf(). Currently not used.

        Returns:
            psf (torch.Tensor): PSF kernels. Shape [ks, ks] or [N, ks, ks].
        """
        points = points.to(self.device)

        # Handle single point vs multiple points
        if len(points.shape) == 1:
            points = points.unsqueeze(0)
            single_point = True
        else:
            single_point = False

        # Calculate circle of confusion for each point
        depths = points[:, 2]  # Shape [N]
        coc_values = self.coc(depths)  # Shape [N]

        # Convert CoC from mm to pixels and add minimum value for numerical stability
        coc_pixel = torch.clamp(
            coc_values / self.pixel_size, min=0.5
        )  # Shape [N], minimum 0.5 pixels
        coc_pixel = (
            coc_pixel.unsqueeze(-1).unsqueeze(-1).repeat(1, ks, ks)
        )  # Shape [N, ks, ks]
        coc_pixel_radius = coc_pixel / 2

        # Create coordinate meshgrid
        x, y = torch.meshgrid(
            torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
            torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        distance_sq = x**2 + y**2

        # Create PSF
        if psf_type == "gaussian":
            # Gaussian PSF
            psf = torch.exp(-distance_sq / (2 * coc_pixel_radius**2)) / (
                2 * np.pi * coc_pixel_radius**2
            )
        elif psf_type == "pillbox":
            # Pillbox PSF
            psf = torch.ones_like(x)
        else:
            raise ValueError(f"Invalid PSF type: {psf_type}")

        # Apply circular mask
        psf_mask = distance_sq < coc_pixel_radius**2
        psf = psf * psf_mask

        # Normalize PSF to sum to 1
        psf = psf / (psf.sum(dim=(-1, -2), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return psf

    def coc(self, depth):
        """Calculate circle of confusion (CoC) [mm].

        Args:
            depth (torch.Tensor): Depth of the object. Shape [B].

        Returns:
            coc (torch.Tensor): Circle of confusion. Shape [B].

        Reference:
            [1] https://en.wikipedia.org/wiki/Circle_of_confusion
        """
        foc_dist = torch.tensor(
            self.foc_dist, device=self.device, dtype=depth.dtype
        ).abs()
        foclen = self.foclen
        fnum = self.fnum

        depth = torch.clamp(depth, self.d_far, self.d_close)
        depth = torch.abs(depth)

        # Calculate circle of confusion diameter, [mm]
        part1 = torch.abs(depth - foc_dist) / depth
        part2 = foclen**2 / (fnum * (foc_dist - foclen))
        coc = part1 * part2

        return coc

    def dof(self, depth):
        """Calculate depth of field [mm].

        Args:
            depth (torch.Tensor): Depth of the object. Shape [B].

        Returns:
            dof (torch.Tensor): Depth of field. Shape [B].

        Reference:
            [1] https://en.wikipedia.org/wiki/Depth_of_field
        """
        depth = torch.clamp(depth, self.d_far, self.d_close)
        depth_abs = torch.abs(depth)

        foclen = self.foclen
        fnum = self.fnum

        # Magnification factor
        m = foclen / (depth_abs - foclen)

        # CoC, [mm]
        coc = self.coc(depth)

        # Depth of field, [mm]
        part1 = 2 * fnum * coc * (m + 1)
        part2 = m**2 - (fnum * coc / foclen) ** 2
        dof = part1 / part2

        return dof

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """Compute RGB PSF."""
        psf = self.psf(points, ks=ks, psf_type="gaussian", **kwargs)
        return psf.unsqueeze(1).repeat(1, 3, 1, 1)

    def psf_map(self, grid=(5, 5), ks=PSF_KS, depth=DEPTH, **kwargs):
        """Compute monochrome PSF map."""
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf = self.psf(points=points, ks=ks, psf_type="gaussian", **kwargs)
        psf_map = psf.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map

    # =============================================
    # Dual-pixel PSF
    # =============================================
    def psf_dp(self, points, ks=PSF_KS):
        """Generate dual-pixel PSF for left and right sub-apertures.

        This function generates separate PSFs for left and right sub-apertures of a dual pixel sensor,
        which enables depth estimation and improved autofocus capabilities.

        Args:
            points (torch.Tensor): Input tensor with shape [N, 3], where columns are [x, y, z] coordinates.
            ks (int): Kernel size for PSF generation.

        Returns:
            tuple: (left_psf, right_psf) where each PSF tensor has shape [N, ks, ks].
        """
        N = points.shape[0]
        depth = points[:, 2]

        # Get the base PSF
        psf_base = self.psf(points, ks=ks, psf_type="gaussian")
        device = psf_base.device

        # Create left and right masks for dual pixel simulation
        l_mask = torch.ones((ks, ks), device=device)
        r_mask = torch.ones((ks, ks), device=device)

        # Split aperture vertically (left half and right half)
        l_pixel, r_pixel = ks // 2, ks // 2 + 1
        l_mask[:, 0:l_pixel] = 0  # Block right side for left PSF
        r_mask[:, r_pixel:] = 0  # Block left side for right PSF

        # Expand masks to match batch dimension [N, ks, ks]
        l_mask = l_mask.unsqueeze(0).repeat(N, 1, 1)
        r_mask = r_mask.unsqueeze(0).repeat(N, 1, 1)

        # Determine focus positions
        depth = depth.to(device)
        foc_dist = torch.tensor(self.foc_dist, device=device, dtype=depth.dtype)
        near_focus_pos = depth > foc_dist  # Shape [N]

        # Create left and right PSFs from base PSF
        psf_l = psf_base.clone()
        psf_r = psf_base.clone()

        # Apply masks based on focus position (this creates the depth-dependent asymmetry)
        # For near focus: left PSF gets left mask, right PSF gets right mask
        # For far focus: masks are swapped to create opposite asymmetry
        for i in range(N):
            if near_focus_pos[i]:
                psf_l[i] = psf_l[i] * l_mask[i]
                psf_r[i] = psf_r[i] * r_mask[i]
            else:
                psf_l[i] = psf_l[i] * r_mask[i]  # Swap masks for far focus
                psf_r[i] = psf_r[i] * l_mask[i]

        # Normalize PSFs
        psf_l = psf_l / (psf_l.sum(dim=(-1, -2), keepdim=True) + EPSILON)
        psf_r = psf_r / (psf_r.sum(dim=(-1, -2), keepdim=True) + EPSILON)

        return psf_l, psf_r

    def psf_rgb_dp(self, points, ks=PSF_KS):
        """Compute RGB dual-pixel PSF."""
        psf_l, psf_r = self.psf_dp(points, ks=ks)
        psf_l = psf_l.unsqueeze(1).repeat(1, 3, 1, 1)
        psf_r = psf_r.unsqueeze(1).repeat(1, 3, 1, 1)
        return psf_l, psf_r

    def psf_map_dp(self, grid=(5, 5), ks=PSF_KS, depth=DEPTH, **kwargs):
        """Compute dual-pixel PSF map."""
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf_l, psf_r = self.psf_dp(points, ks=ks, **kwargs)
        psf_map_l = psf_l.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        psf_map_r = psf_r.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map_l, psf_map_r

    def render_rgbd_dp(self, rgb_img, depth):
        """Render RGBD image with dual-pixel PSF.

        Args:
            rgb_img (tensor): [B, 3, H, W]
            depth (tensor): [B, 1, H, W]

        Returns:
            img_left (tensor): [B, 3, H, W]
            img_right (tensor): [B, 3, H, W]
        """
        # Convert depth to negative values
        if (depth > 0).any():
            depth = -depth

        depth_min = depth.min()
        depth_max = depth.max()
        num_depth = 10
        psf_center = (0.0, 0.0)
        psf_ks = PSF_KS

        # Calculate dual-pixel PSF at reference depths
        depths_ref = torch.linspace(depth_min, depth_max, num_depth).to(self.device)
        points = torch.stack(
            [
                torch.full_like(depths_ref, psf_center[0]),
                torch.full_like(depths_ref, psf_center[1]),
                depths_ref,
            ],
            dim=-1,
        )
        psfs_left, psfs_right = self.psf_rgb_dp(
            points=points, ks=psf_ks
        )  # shape [num_depth, 3, ks, ks]

        # Render dual-pixel image with PSF convolution and depth interpolation
        img_left = conv_psf_depth_interp(rgb_img, depth, psfs_left, depths_ref)
        img_right = conv_psf_depth_interp(rgb_img, depth, psfs_right, depths_ref)
        return img_left, img_right


if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image

    lens = ParaxialLens(
        foclen=50, fnum=1.8, sensor_size=(20.0, 20.0), sensor_res=(2000, 2000)
    )
    lens.refocus(-1000)
    lens.draw_psf_map(
        save_name="./psf_map_paraxial_depth1500_focus1000.png",
        grid=(11, 11),
        ks=PSF_KS,
        depth=-1500,
        log_scale=False,
    )

    # PSF DP far
    psf_map_l, psf_map_r = lens.psf_map_dp(grid=(11, 11), ks=128, depth=-1500)
    psf_map_l = psf_map_l.reshape(-1, 1, 128, 128)
    psf_map_r = psf_map_r.reshape(-1, 1, 128, 128)
    save_image(
        make_grid(psf_map_l, nrow=11),
        "./psf_map_dp_left_depth1500_focus1000.png",
        normalize=True,
    )
    save_image(
        make_grid(psf_map_r, nrow=11),
        "./psf_map_dp_right_depth1500_focus1000.png",
        normalize=True,
    )

    # PSF DP near
    psf_map_l, psf_map_r = lens.psf_map_dp(grid=(11, 11), ks=128, depth=-800)
    psf_map_l = psf_map_l.reshape(-1, 1, 128, 128)
    psf_map_r = psf_map_r.reshape(-1, 1, 128, 128)
    save_image(
        make_grid(psf_map_l, nrow=11),
        "./psf_map_dp_left_depth800_focus1000.png",
        normalize=True,
    )
    save_image(
        make_grid(psf_map_r, nrow=11),
        "./psf_map_dp_right_depth800_focus1000.png",
        normalize=True,
    )

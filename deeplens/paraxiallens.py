# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Paraxial ABCD matrix geometric lens model.

Paraxial lens can model defocus (Circle of confusion) but not aberration. It is used in some renderers like Blender, Photoshop, etc.

Reference:
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
"""

import numpy as np
import torch

from deeplens.lens import Lens
from deeplens.optics.basics import DEPTH
from deeplens.optics.psf import conv_psf_pixel


# ==================================================================
# Paraxial lens
# ==================================================================
class ParaxialLens(Lens):
    def __init__(self, foclen, fnum, sensor_size, sensor_res, device="cpu"):
        super(ParaxialLens, self).__init__()

        # Lens parameters
        self.foclen = foclen
        self.fnum = fnum

        # Sensor size and resolution
        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]  # Pixel size [mm]

        self.d_far = -20000.0
        self.d_close = -200.0

    def refocus(self, foc_dist):
        """Refocus the lens to the given focus distance."""
        self.foc_dist = foc_dist

    def psf_rgb(self, points, ks=51, **kwargs):
        """Compute RGB PSF."""
        psf = self.psf(points, ks=ks, psf_type="gaussian", **kwargs)
        return psf.unsqueeze(1).repeat(1, 3, 1, 1)

    def psf_map(self, grid=(5, 5), ks=51, depth=DEPTH, **kwargs):
        """Compute monochrome PSF map."""
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf = self.psf(points=points, ks=ks, psf_type="gaussian", **kwargs)
        psf_map = psf.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map

    def psf(self, points, ks=51, psf_type="gaussian", **kwargs):
        """PSF is modeled as a 2D uniform circular disk with diameter CoC.

        Args:
            points (torch.Tensor): Points of the object. Shape [N, 3] or [3].
            ks (int): Kernel size.
            psf_type (str): PSF type. "gaussian" or "pillbox".
            **kwargs: Additional arguments for psf(). Currently not used.

        Returns:
            psf (torch.Tensor): PSF kernels. Shape [ks, ks] or [N, ks, ks].
        """
        device = self.device
        points = points.to(device)

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
        x, y = x.to(device), y.to(device)
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
        psf = psf / (psf.sum(dim=(-1, -2), keepdim=True) + 1e-8)

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
            self.foc_dist, device=depth.device, dtype=depth.dtype
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

    def render(self, img, depth, foc_dist, high_res=False, psf_ks=51):
        """Render image with aif image and PSF.

        Args:
            img: [N, C, H, W]
            depth: [N, 1, H, W]
            foc_dist: [N]

        Raises:
            Exception: Untested.

        Returns:
            render (torch.Tensor): Rendered image. Shape [N, C, H, W].
        """
        ks = psf_ks
        device = img.device

        if len(img.shape) == 3:
            raise Exception("Untested.")

        elif len(img.shape) == 4:
            N, C, H, W = img.shape

            # [N] to [N, 1, H, W]
            foc_dist = (
                foc_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            )

            psf = torch.zeros((N, H, W, ks, ks), device=device)
            x, y = torch.meshgrid(
                torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
                torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
                indexing="xy",
            )
            x, y = x.to(device), y.to(device)

            coc_pixel = self.coc(depth, foc_dist)
            # Shape expands to [N, H, W, ks, ks]
            coc_pixel = (
                coc_pixel.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, ks, ks)
            )
            coc_pixel_radius = coc_pixel / 2
            psf = torch.exp(-(x**2 + y**2) / 2 / coc_pixel_radius**2) / (
                2 * np.pi * coc_pixel_radius**2
            )
            psf_mask = x**2 + y**2 < coc_pixel_radius**2
            psf = psf * psf_mask
            psf = psf / psf.sum((-1, -2)).unsqueeze(-1).unsqueeze(-1)

            render = conv_psf_pixel(img, psf)
            return render

    # =============================================
    # Dual-pixel PSF
    # =============================================
    def psf_dp(self, points, ks=51):
        """Generate dual-pixel PSF for left and right sub-apertures.

        This function generates separate PSFs for left and right sub-apertures of a dual pixel sensor,
        which enables depth estimation and improved autofocus capabilities.

        Args:
            points (torch.Tensor): Input tensor with shape [N, 3], where columns are [x, y, z] coordinates.
            ks (int): Kernel size for PSF generation.

        Returns:
            tuple: (left_psf, right_psf) where each PSF tensor has shape [N, ks, ks].
        """
        # Extract depth information
        depth = points[:, 2]  # Shape [N]

        # Get the base PSF using the existing psf() function
        psf_base = self.psf(points, ks=ks, psf_type="gaussian")
        device = psf_base.device

        N = psf_base.shape[0]

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
        depth = depth.to(device)  # Ensure depth is on the correct device
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

        # Normalize PSFs separately
        psf_l = psf_l / (psf_l.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        psf_r = psf_r / (psf_r.sum(dim=(-1, -2), keepdim=True) + 1e-8)

        return psf_l, psf_r

    def psf_map_dp(self, grid=(5, 5), ks=51, depth=DEPTH, **kwargs):
        """Compute dual-pixel PSF map."""
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf_l, psf_r = self.psf_dp(points, ks=ks, **kwargs)
        psf_map_l = psf_l.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        psf_map_r = psf_r.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map_l, psf_map_r


if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image

    lens = ParaxialLens(
        foclen=50, fnum=1.8, sensor_size=(20.0, 20.0), sensor_res=(2000, 2000)
    )
    lens.refocus(-1000)
    lens.draw_psf_map(
        save_name="./psf_map_paraxial_depth1500_focus1000.png",
        grid=(11, 11),
        ks=128,
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

# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Optimization and constraint functions for GeoLens.

Differentiable lens design is typically better than conventional optimization methods for several reasons:
    1. AutoDiff calculates more accurate gradients, which is important for complex optical systems.
    2. First-order optimization methods are more stable than second-order methods (e.g., Levenberg-Marquardt).
    3. Efficient definition of loss functions on lens design constraints.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deeplens.optics.basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from deeplens.utils import set_logger


class GeoLensOptim:
    """This class contains the optimization functions for the geometric lens design."""

    # ================================================================
    # Lens design constraints
    # ================================================================
    def init_constraints(self):
        """Initialize constraints for the lens design. Unit [mm]."""
        if self.r_sensor < 12.0:
            self.is_cellphone = True

            # Self intersection constraints
            self.dist_min = 0.05
            self.dist_max = 1.0
            self.thickness_min = 0.25
            self.thickness_max = 2.0
            self.flange_min = 0.25
            self.flange_max = 3.0

            # Surface curvature constraints
            self.sag_max = 2.0
            self.grad_max = 1.0
            self.grad2_max = 100.0
            self.d_to_t_max = 10.0
            self.tmax_to_tmin_max = 5.0

            # Chief ray angle constraints
            self.chief_ray_angle_max = 20.0
        else:
            self.is_cellphone = False

            # Self-intersection constraints
            self.dist_min = 0.1
            self.dist_max = 50.0  # float("inf")
            self.thickness_min = 0.3
            self.thickness_max = 50.0  # float("inf")
            self.flange_min = 0.5
            self.flange_max = 50.0  # float("inf")

            # Surface curvature constraints
            self.sag_max = 10.0
            self.grad_max = 1.0
            self.grad2_max = 100.0
            self.d_to_t_max = 10.0
            self.tmax_to_tmin_max = 5.0

            # Chief ray angle constraints
            self.chief_ray_angle_max = 20.0

    def loss_reg(self, w_focus=1.0, w_intersec=2.0, w_surf=1.0, w_chief_ray_angle=5.0):
        """An empirical regularization loss for lens design.

        By default we should use weight 0.1 * self.loss_reg() in the total loss.
        """
        # Loss functions for regularization
        loss_focus = self.loss_infocus()
        loss_intersec = self.loss_self_intersec()
        loss_surf = self.loss_surface()
        loss_chief_ray_angle = self.loss_chief_ray_angle()
        loss_reg = (
            w_focus * loss_focus
            + w_intersec * loss_intersec
            + w_surf * loss_surf
            + w_chief_ray_angle * loss_chief_ray_angle
        )

        # Return loss and loss dictionary
        loss_dict = {
            "loss_focus": loss_focus.item(),
            "loss_intersec": loss_intersec.item(),
            "loss_surf": loss_surf.item(),
            'loss_chief_ray_angle': loss_chief_ray_angle.item(),
        }
        return loss_reg, loss_dict

    def loss_infocus(self, target=0.01):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            target (float, optional): target of RMS loss. Defaults to 0.005 [mm].
        """
        loss = torch.tensor(0.0, device=self.device)

        # Ray tracing and calculate RMS error
        ray = self.sample_parallel(fov_x=0.0, fov_y=0.0, wvln=WAVE_RGB[1])
        ray = self.trace2sensor(ray)
        rms_error = ray.rms_error()

        # If RMS error is larger than target, add it to loss
        if rms_error > target:
            loss += rms_error

        return loss

    def loss_surface(self):
        """Penalize surface shape:
        1. Penalize maximum sag
        2. Penalize diameter to thickness ratio
        3. Penalize thick_max to thick_min ratio
        4. Penalize diameter to thickness ratio
        5. Penalize maximum to minimum thickness ratio
        """
        sag_max_allowed = self.sag_max
        grad_max_allowed = self.grad_max
        grad2_max_allowed = self.grad2_max
        d_to_t_max = self.d_to_t_max
        tmax_to_tmin_max = self.tmax_to_tmin_max

        loss = torch.tensor(0.0, device=self.device)
        loss_d_to_t = torch.tensor(0.0, device=self.device)
        loss_tmax_to_tmin = torch.tensor(0.0, device=self.device)
        for i in self.find_diff_surf():
            # Sample points on the surface
            x_ls = torch.linspace(0.0, 1.0, 20).to(self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag_max = sag_ls.abs().max()
            if sag_max > sag_max_allowed:
                loss += sag_max

            # 1st-order derivative
            grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            grad_max = grad_ls.abs().max()
            if grad_max > grad_max_allowed:
                loss += 10 * grad_max

            # 2nd-order derivative
            grad2_ls = self.surfaces[i].d2fdxyz2(x_ls, y_ls)[0]
            grad2_max = grad2_ls.abs().max()
            if grad2_max > grad2_max_allowed:
                loss += 10 * grad2_max

            # Diameter to thickness ratio, thick_max to thick_min ratio
            if not self.surfaces[i].mat2.name == "air":
                surf2 = self.surfaces[i + 1]
                surf1 = self.surfaces[i]

                # Penalize diameter to thickness ratio
                d_to_t = max(surf2.r, surf1.r) / (surf2.d - surf1.d)
                if d_to_t > d_to_t_max:
                    loss_d_to_t += d_to_t

                # Penalize thick_max to thick_min ratio
                r_edge = min(surf2.r, surf1.r)
                thick_center = surf2.d - surf1.d
                thick_edge = surf2.surface_with_offset(r_edge, 0.0) - surf1.surface_with_offset(r_edge, 0.0)
                if thick_center > thick_edge:
                    tmax_to_tmin = thick_center / thick_edge
                else:
                    tmax_to_tmin = thick_edge / thick_center

                if tmax_to_tmin > tmax_to_tmin_max:
                    loss_tmax_to_tmin += tmax_to_tmin

        return loss + loss_d_to_t + loss_tmax_to_tmin

    def loss_self_intersec(self):
        """Loss function to avoid self-intersection.

        Loss is designed by the distance to the next surfaces.
        """
        # Constraints
        space_min_allowed = self.dist_min
        space_max_allowed = self.dist_max
        thickness_min_allowed = self.thickness_min
        thickness_max_allowed = self.thickness_max
        flange_min_allowed = self.flange_min
        flange_max_allowed = self.flange_max

        loss_min = torch.tensor(0.0, device=self.device)
        loss_max = torch.tensor(0.0, device=self.device)

        # Distance between surfaces
        for i in range(len(self.surfaces) - 1):
            # Sample points on the two surfaces
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]
            r = torch.linspace(0.0, 1.0, 20).to(self.device) * current_surf.r
            z_front = current_surf.surface_with_offset(r, 0.0)
            z_next = next_surf.surface_with_offset(r, 0.0)

            # Air gap
            if self.surfaces[i].mat2.name == "air":
                # Constrain minimum distance between surfaces
                dist_min = torch.min(z_next - z_front)
                if dist_min < space_min_allowed:
                    loss_min += dist_min

                # Constrain maximum center distance between surfaces
                dist_max = z_next[0] - z_front[0]
                if dist_max > space_max_allowed:
                    loss_max += dist_max

            # Lens thickness
            else:
                # Constrain minimum distance of elements
                dist_min = torch.min(z_next - z_front)
                if dist_min < thickness_min_allowed:
                    loss_min += dist_min

                # Constrain maximum distance of elements
                dist_max = torch.max(z_next - z_front)
                if dist_max > thickness_max_allowed:
                    loss_max += dist_max

        # Distance to sensor (flange)
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 20).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        flange = torch.min(z_last_surf)
        if flange < flange_min_allowed:
            loss_min += flange
        if flange > flange_max_allowed:
            loss_max += flange

        # Loss, minimize loss_max and maximize loss_min
        return loss_max - loss_min

    def loss_ray_angle(self, target=0.5, depth=DEPTH):
        """Loss function designed to penalize large incident angle rays.

        Oblique angle is defined as the cosine of the angle between the ray and the normal vector of the surface.

        Args:
            target (float, optional): target of ray angle. Defaults to 0.5.
            depth (float, optional): depth of the lens. Defaults to DEPTH.
        """
        # Sample grid rays, shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(num_grid=GEO_GRID, depth=depth, sample_more_off_axis=True)
        ray = self.trace2sensor(ray)

        # Loss (we want to maximize ray angle term)
        mask = ray.obliq < target
        if mask.any():
            loss = ray.obliq[mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device)

        # We want to maximize ray angle term
        return -loss

    def loss_chief_ray_angle(self):
        """Chief ray angle loss function."""
        max_angle_deg = self.chief_ray_angle_max

        # Ray tracing
        ray = self.sample_grid_rays(num_grid=GEO_GRID, num_rays=SPP_CALC, scale_pupil=0.25)
        ray = self.trace2sensor(ray)

        # Calculate chief ray angle
        cos_cra = ray.d[..., 2]
        cos_cra_ref = float(np.cos(np.deg2rad(max_angle_deg)))
        cos_cra = torch.where(
            cos_cra < cos_cra_ref,
            cos_cra,
            torch.tensor(cos_cra_ref, device=self.device),
        )

        # Loss
        loss = -cos_cra.mean()

        return loss

    # ================================================================
    # Loss functions for image quality
    # ================================================================
    def loss_rms(
        self,
        num_grid=GEO_GRID,
        depth=DEPTH,
        num_rays=SPP_PSF,
        sample_more_off_axis=False,
    ):
        """Compute average RMS errors. Baseline RMS loss function.

        Compared to the loss function developed in the paper, this loss function doesnot have a weight mask.

        Args:
            num_grid (int, optional): Number of grid points. Defaults to GEO_GRID.
            depth (float, optional): Depth of the lens. Defaults to DEPTH.
            num_rays (int, optional): Number of rays. Defaults to SPP_CALC.
            sample_more_off_axis (bool, optional): Whether to sample more off-axis rays. Defaults to False.

        Returns:
            avg_rms_error (torch.Tensor): RMS error averaged over wavelengths and grid points.
        """
        all_rms_errors = []
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            ray = self.sample_grid_rays(
                num_grid=num_grid,
                depth=depth,
                num_rays=num_rays,
                wvln=wvln,
                sample_more_off_axis=sample_more_off_axis,
            )

            # Calculate reference center, shape of (..., 2)
            if i == 0:
                with torch.no_grad():
                    ray_center_green = -self.psf_center(point=ray.o[:, :, 0, :], method="pinhole")

            ray = self.trace2sensor(ray)

            # # Green light point center for reference
            # if i == 0:
            #     with torch.no_grad():
            #         ray_center_green = ray.centroid()

            # Calculate RMS error with reference center
            rms_error = ray.rms_error(center_ref=ray_center_green)
            all_rms_errors.append(rms_error)

        # Calculate average RMS error
        avg_rms_error = torch.stack(all_rms_errors).mean(dim=0)
        return avg_rms_error

    def loss_rms_infinite(self, num_grid=GEO_GRID, depth=DEPTH, num_rays=SPP_CALC):
        """Compute RGB RMS error per pixel using Zernike polynomials.

        Args:
            num_fields: Number of fields. Defaults to 3.
            depth: object space depth. Defaults to DEPTH.
        """
        # calculate fov_x and fov_y
        [H, W] = self.sensor_res
        tan_fov_y = np.sqrt(np.tan(self.hfov) ** 2 / (1 + W**2 / H**2))
        tan_fov_x = np.sqrt(np.tan(self.hfov) ** 2 - tan_fov_y**2)
        fov_y = np.rad2deg(np.arctan(tan_fov_y))
        fov_x = np.rad2deg(np.arctan(tan_fov_x))
        fov_y = torch.linspace(0.0, fov_y, num_grid).tolist()
        fov_x = torch.linspace(0.0, fov_x, num_grid).tolist()

        # calculate RMS error
        all_rms_errors = []
        all_rms_radii = []
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Ray tracing
            ray = self.sample_parallel(fov_x=fov_x, fov_y=fov_y, num_rays=num_rays, wvln=wvln, depth=depth)
            ray = self.trace2sensor(ray)

            # Green light point center for reference
            if i == 0:
                pointc_green = (ray.o[..., :2] * ray.valid.unsqueeze(-1)).sum(-2) / ray.valid.sum(-1).add(EPSILON).unsqueeze(-1)  # shape [1, num_fields, 2]
                pointc_green = pointc_green.unsqueeze(-2).repeat(1, 1, SPP_PSF, 1)  # shape [num_fields, num_fields, num_rays, 2]

            # Calculate RMS error for different FoVs
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.valid.unsqueeze(-1)

            # error
            rms_error = torch.mean((((o2_norm**2).sum(-1) * ray.valid).sum(-1) / (ray.valid.sum(-1) + EPSILON)).sqrt())

            # radius
            rms_radius = torch.mean(((o2_norm**2).sum(-1) * ray.valid).sqrt().max(dim=-1).values)
            all_rms_errors.append(rms_error)
            all_rms_radii.append(rms_radius)

        # Calculate and print average across wavelengths
        avg_rms_error = torch.stack(all_rms_errors).mean(dim=0)
        avg_rms_radius = torch.stack(all_rms_radii).mean(dim=0)

        return avg_rms_error

    # def loss_mtf(self, relative_fov=[0.0, 0.7, 1.0], depth=DEPTH, wvln=DEFAULT_WAVE):
    #     """Loss function designed on the MTF. We want to maximize MTF values."""
    #     loss = 0.0
    #     for fov in relative_fov:
    #         # ==> Calculate PSF
    #         point = torch.tensor([fov, fov, depth])
    #         psf = self.psf(points=point, wvln=wvln, ks=256)

    #         # ==> Calculate MTF
    #         x = torch.linspace(-1, 1, psf.shape[1]) * self.pixel_size * psf.shape[1] / 2
    #         y = torch.linspace(-1, 1, psf.shape[0]) * self.pixel_size * psf.shape[0] / 2

    #         # Extract 1D PSFs along the sagittal and tangential directions
    #         center_x = psf.shape[1] // 2
    #         center_y = psf.shape[0] // 2
    #         sagittal_psf = psf[center_y, :]
    #         tangential_psf = psf[:, center_x]

    #         # Fourier Transform to get the MTFs
    #         sagittal_mtf = torch.abs(torch.fft.fft(sagittal_psf))
    #         tangential_mtf = torch.abs(torch.fft.fft(tangential_psf))

    #         # Normalize the MTFs
    #         sagittal_mtf /= sagittal_mtf.max().detach()
    #         tangential_mtf /= tangential_mtf.max().detach()
    #         delta_x = self.pixel_size

    #         # Create frequency axis in cycles/mm
    #         freq = np.fft.fftfreq(psf.shape[0], delta_x)

    #         # Only keep the positive frequencies
    #         positive_freq_idx = freq > 0

    #         loss += torch.sum(
    #             sagittal_mtf[positive_freq_idx] + tangential_mtf[positive_freq_idx]
    #         ) / len(positive_freq_idx)

    #     return -loss

    # ================================================================
    # Example optimization function
    # ================================================================
    def sample_ring_arm_rays(self, num_ring=10, num_arm=10, spp=1000, depth=DEPTH, wvln=DEFAULT_WAVE, scale_pupil=1.0):
        """Sample rays from object space using a ring-arm pattern.

        This method distributes sampling points (origins of ray bundles) on a polar grid in the object plane,
        defined by field of view. This is useful for capturing lens performance across the full field.
        The points include the center and `num_ring` rings with `num_arm` points on each.

        Args:
            num_ring (int): Number of rings to sample in the field of view.
            num_arm (int): Number of arms (spokes) to sample for each ring.
            spp (int): Total number of rays to be sampled, distributed among field points.
            depth (float): Depth of the object plane.
            wvln (float): Wavelength of the rays.
            scale_pupil (float): Scale factor for the pupil size.

        Returns:
            Ray: A Ray object containing the sampled rays.
        """
        # Create points on rings and arms
        max_fov_rad = self.hfov
        ring_fovs = max_fov_rad * torch.sqrt(torch.linspace(0.0, 1.0, num_ring, device=self.device))
        arm_angles = torch.linspace(0.0, 2 * np.pi, num_arm + 1, device=self.device)[:-1]
        ring_grid, arm_grid = torch.meshgrid(ring_fovs, arm_angles, indexing="ij")
        fov_x_rad = ring_grid * torch.cos(arm_grid)
        fov_y_rad = ring_grid * torch.sin(arm_grid)
        x = depth * torch.tan(fov_x_rad)
        y = depth * torch.tan(fov_y_rad)
        z = torch.full_like(x, depth)
        points = torch.stack([x, y, z], dim=-1)  # shape: [num_ring, num_arm, 3]

        # Sample rays
        rays = self.sample_from_points(points=points, num_rays=spp, wvln=wvln, scale_pupil=scale_pupil)
        return rays

    def optimize(
        self,
        lrs=[1e-4, 1e-4, 1e-1, 1e-4],
        decay=0.01,
        iterations=5000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        result_dir=None,
    ):
        """Optimize the lens by minimizing rms errors.

        Debug hints:
            *, Slowly and continuously update!
            1, thickness (fov and ttl should match)
            2, alpha order (higher is better but more sensitive)
            3, learning rate and decay (prefer smaller lr and decay)
            4, correct params range
            5. nan can be introduced by torch.sqrt() function in the backward pass.
        """
        # Experiment settings
        depth = DEPTH
        num_ring = 10
        num_arm = 10
        spp = 1024
        sample_rays_per_iter = 5 * test_per_iter if centroid else test_per_iter

        # Result directory and logger
        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-DesignLens"

        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(f"lr:{lrs}, iterations:{iterations}, num_ring:{num_ring}, num_arm:{num_arm}, rays_per_fov:{spp}.")

        # Optimizer and scheduler
        optimizer = self.get_optimizer(lrs, decay=decay, optim_mat=optim_mat)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=iterations)

        # Training loop
        pbar = tqdm(
            total=iterations + 1,
            desc="Progress",
            postfix={"loss_rms": 0},
        )
        for i in range(iterations + 1):
            # ===> Evaluate the lens
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if shape_control:
                        self.correct_shape()

                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}")

            # ===> Sample new rays and calculate center
            if i % sample_rays_per_iter == 0:
                with torch.no_grad():
                    # Sample rays
                    self.update_float_setting()
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.sample_ring_arm_rays(num_ring=num_ring, num_arm=num_arm, spp=spp, depth=depth, wvln=wv, scale_pupil=1.0)
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_ref = -self.psf_center(point=ray.o[:, :, 0, :], method="chief_ray")
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)
                    else:
                        center_ref = -self.psf_center(point=ray.o[:, :, 0, :], method="pinhole")
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

            # ===> Optimize lens by minimizing RMS
            loss_rms_ls = []
            for wv_idx, wv in enumerate(WAVE_RGB):
                # Ray tracing to sensor, [num_grid, num_grid, num_rays, 3]
                ray = rays_backup[wv_idx].clone()
                ray = self.trace2sensor(ray)

                # Ray error to center and valid mask
                ray_xy = ray.o[..., :2]
                ray_valid = ray.valid
                ray_err = ray_xy - center_ref

                # Weight mask, shape of [num_grid, num_grid]
                if wv_idx == 0:
                    with torch.no_grad():
                        weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                        weight_mask /= ray_valid.sum(-1) + EPSILON
                        weight_mask /= weight_mask.mean()

                # Loss on RMS error
                l_rms = (((ray_err**2).sum(-1) + EPSILON).sqrt() * ray_valid).sum(-1)
                l_rms /= ray_valid.sum(-1) + EPSILON

                # Weighted loss
                l_rms_weighted = (l_rms * weight_mask).sum()
                l_rms_weighted /= weight_mask.sum() + EPSILON
                loss_rms_ls.append(l_rms_weighted)

            loss_rms = sum(loss_rms_ls) / len(loss_rms_ls)

            # Total loss
            loss_reg, loss_dict = self.loss_reg()
            w_reg = 0.05
            L_total = loss_rms + w_reg * loss_reg

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
            pbar.update(1)

        pbar.close()

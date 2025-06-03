"""Optimization and constraint functions for GeoLens.

Differentiable lens design is typically better than conventional optimization methods for several reasons:
    1. AutoDiff calculates more accurate gradients, which is important for complex optical systems.
    2. First-order optimization methods are more stable than second-order methods (e.g., Levenberg-Marquardt).
    3. Efficient definition of loss functions on lens design constraints.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.

Copyright (c) 2025 Xinge Yang (xinge.yang@kaust.edu.sa)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
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
    # ================================================================
    # Lens design constraints
    # ================================================================
    def init_constraints(self):
        """Initialize constraints for the lens design. Unit [mm]."""
        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.dist_min = 0.05
            self.dist_max = 0.6
            self.thickness_min = 0.2
            self.thickness_max = 1.5
            self.flange_min = 0.25
            self.flange_max = 3.0

            self.sag_max = 2.0
            self.grad_max = 1.0
            self.grad2_max = 100.0
        else:
            self.is_cellphone = False

            self.dist_min = 0.1
            self.dist_max = 50.0 #float("inf")
            self.thickness_min = 0.3
            self.thickness_max = 50.0 #float("inf")
            self.flange_min = 0.5
            self.flange_max = 50.0 #float("inf")

            self.sag_max = 10.0
            self.grad_max = 1.0
            self.grad2_max = 100.0

    # ================================================================
    # Lens design loss functions
    # ================================================================
    def loss_reg(self, w_focus=None):
        """An empirical regularization loss for lens design. By default we should use weight 0.1 * self.loss_reg() in the total loss."""
        loss_focus = self.loss_infocus()

        if self.is_cellphone:
            loss_intersec = self.loss_self_intersec()
            loss_surf = self.loss_surface()
            # loss_angle = self.loss_ray_angle()

            w_focus = 2.0 if w_focus is None else w_focus
            loss_reg = (
                w_focus * loss_focus + 1.0 * loss_intersec 
                + 1.0 * loss_surf 
                # + 0.1 * loss_angle
            )
        else:
            loss_intersec = self.loss_self_intersec()
            loss_surf = self.loss_surface()
            # loss_angle = self.loss_ray_angle()

            w_focus = 5.0 if w_focus is None else w_focus
            loss_reg = (
                w_focus * loss_focus + 1.0 * loss_intersec 
                + 1.0 * loss_surf 
                # + 0.05 * loss_angle
            )

        return loss_reg

    def loss_infocus(self, target=0.005):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            target (float, optional): target of RMS loss. Defaults to 0.005 [mm].
        """
        loss = torch.tensor(0.0, device=self.device)
        for wv in WAVE_RGB:
            # Ray tracing and calculate RMS error
            ray = self.sample_parallel(fov_x=0.0, fov_y=0.0, wvln=wv)
            ray = self.trace2sensor(ray)
            rms_error = ray.rms_error()

            # If RMS error is larger than target, add it to loss
            if rms_error > target:
                loss += rms_error

        return loss / len(WAVE_RGB)

    def loss_rms(
        self,
        num_grid=GEO_GRID,
        depth=DEPTH,
        num_rays=SPP_CALC,
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
            ray = self.trace2sensor(ray)

            # Green light point center for reference
            if i == 0:
                with torch.no_grad():
                    ray_center_green = ray.centroid()

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
            ray = self.sample_parallel(
                fov_x=fov_x, fov_y=fov_y, num_rays=num_rays, wvln=wvln, depth=depth
            )
            ray = self.trace2sensor(ray)

            # Green light point center for reference
            if i == 0:
                pointc_green = (ray.o[..., :2] * ray.valid.unsqueeze(-1)).sum(
                    -2
                ) / ray.valid.sum(-1).add(EPSILON).unsqueeze(
                    -1
                )  # shape [1, num_fields, 2]
                pointc_green = pointc_green.unsqueeze(-2).repeat(
                    1, 1, SPP_PSF, 1
                )  # shape [num_fields, num_fields, num_rays, 2]

            # Calculate RMS error for different FoVs
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.valid.unsqueeze(-1)

            # error
            rms_error = torch.mean(
                (
                    ((o2_norm**2).sum(-1) * ray.valid).sum(-1) / (ray.valid.sum(-1) + EPSILON)
                ).sqrt()
            )

            # radius
            rms_radius = torch.mean(
                ((o2_norm**2).sum(-1) * ray.valid).sqrt().max(dim=-1).values
            )
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

    def loss_surface(self):
        """Penalize surface to prevent surface from being too curved.

        Loss is designed by the maximum sag, first-order derivative, and second-order derivative.
        """
        sag_max_allowed = self.sag_max
        grad_max_allowed = self.grad_max
        grad2_max_allowed = self.grad2_max

        loss = torch.tensor(0.0, device=self.device)
        for i in self.find_diff_surf():
            # Sample points on the surface
            x_ls = torch.linspace(0.0, 1.0, 20).to(self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag_max = sag_ls.abs().max()
            if sag_max > sag_max_allowed:
                loss += sag_max

            # # 1st-order derivative
            # grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            # grad_max = grad_ls.abs().max()
            # if grad_max > grad_max_allowed:
            #     loss += 10 * grad_max

            # # 2nd-order derivative
            # grad2_ls = self.surfaces[i].d2fdxyz2(x_ls, y_ls)[0]
            # grad2_max = grad2_ls.abs().max()
            # if grad2_max > grad2_max_allowed:
            #     loss += 10 * grad2_max

        return loss

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

        # Constraints for distance between surfaces
        for i in range(len(self.surfaces) - 1):
            # Sample points on the two surfaces
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]
            r = torch.linspace(0.0, 1.0, 20).to(self.device) * current_surf.r
            z_front = current_surf.surface_with_offset(r, 0.0)
            z_next = next_surf.surface_with_offset(r, 0.0)

            # Minimum and maximum distance between surfaces
            dist_min = torch.min(z_next - z_front)
            dist_max = torch.max(z_next - z_front)
            if self.surfaces[i].mat2.name == "air":
                if dist_min < space_min_allowed:
                    loss_min += dist_min
                if dist_max > space_max_allowed:
                    loss_max += dist_max
            else:
                if dist_min < thickness_min_allowed:
                    loss_min += dist_min
                if dist_max > thickness_max_allowed:
                    loss_max += dist_max

        # Constraints for distance to sensor (flange distance)
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 20).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        flange_min = torch.min(z_last_surf)
        flange_max = torch.max(z_last_surf)
        if flange_min < flange_min_allowed:
            loss_min += flange_min
        if flange_max > flange_max_allowed:
            loss_max += flange_max

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
        ray = self.sample_grid_rays(
            num_grid=GEO_GRID, depth=depth, sample_more_off_axis=True
        )
        ray = self.trace2sensor(ray)

        # Loss (we want to maximize ray angle term)
        mask = ray.obliq < target
        if mask.any():
            loss = ray.obliq[mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device)

        # We want to maximize ray angle term
        return -loss

    # ================================================================
    # Example optimization function
    # ================================================================
    def optimize(
        self,
        lrs=[1e-3, 1e-4, 1e-1, 1e-4],
        decay=0.001,
        iterations=2000,
        test_per_iter=50,
        centroid=False,
        optim_mat=False,
        match_mat=False,
        shape_control=True,
        sample_more_off_axis=False,
        result_dir="./results",
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
        # Preparation
        depth = DEPTH
        num_grid = 41
        spp = 512

        sample_rays_per_iter = 5 * test_per_iter if centroid else test_per_iter

        result_dir = (
            result_dir + "/" + datetime.now().strftime("%m%d-%H%M%S") + "-DesignLens"
        )
        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(
            f"lr:{lrs}, decay:{decay}, iterations:{iterations}, spp:{spp}, grid:{num_grid}."
        )

        optimizer = self.get_optimizer(lrs, decay, optim_mat=optim_mat)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=200, num_training_steps=iterations
        )

        # Training loop
        pbar = tqdm(
            total=iterations + 1, desc="Progress", postfix={"loss_rms": 0, "loss_reg": 0}
        )
        for i in range(iterations + 1):
            # ===> Evaluate the lens
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if i > 0:
                        if shape_control:
                            self.correct_shape()

                        if optim_mat and match_mat:
                            self.match_materials()

                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(
                        f"{result_dir}/iter{i}",
                        multi_plot=False,
                    )

            # ===> Sample new rays and calculate center
            if i % sample_rays_per_iter == 0:
                with torch.no_grad():
                    # Sample rays
                    self.update_computation()
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.sample_grid_rays(
                            num_grid=num_grid,
                            depth=depth,
                            num_rays=spp,
                            wvln=wv,
                            sample_more_off_axis=sample_more_off_axis,
                        )
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_ref = -self.psf_center(
                            point=ray.o[:, :, 0, :], method="chief_ray"
                        )
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)
                    else:
                        center_ref = -self.psf_center(
                            point=ray.o[:, :, 0, :], method="pinhole"
                        )
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

                # # Use only quater of the sensor
                # ray_err = ray_err[num_grid // 2 :, num_grid // 2 :, :, :]
                # ray_valid = ray_valid[num_grid // 2 :, num_grid // 2 :, :]

                # Weight mask, shape of [num_grid, num_grid]
                if wv_idx == 0:
                    with torch.no_grad():
                        weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                        weight_mask /= ray_valid.sum(-1) + EPSILON
                        weight_mask = weight_mask.sqrt()
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
            loss_reg = self.loss_reg()
            w_reg = 0.05
            L_total = loss_rms + w_reg * loss_reg

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss_rms=loss_rms.item(), loss_reg=loss_reg.item())
            pbar.update(1)

        pbar.close()

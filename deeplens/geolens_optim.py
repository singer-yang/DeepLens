"""Optimization functions for GeoLens."""

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
    PSF_KS,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from deeplens.utils import set_logger


class GeoLensOptim:
    # ================================================================
    # Constraints
    # ================================================================
    def init_constraints(self):
        """Initialize constraints for the lens design."""
        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.dist_min = 0.1
            self.dist_max = 0.6  # float("inf")
            self.thickness_min = 0.3
            self.thickness_max = 1.5
            self.flange_min = 0.5
            self.flange_max = 3.0  # float("inf")

            self.sag_max = 0.8
            self.grad_max = 1.0
            self.grad2_max = 100.0
        else:
            self.is_cellphone = False

            self.dist_min = 0.1
            self.dist_max = float("inf")
            self.thickness_min = 0.3
            self.thickness_max = float("inf")
            self.flange_min = 0.5
            self.flange_max = 50.0  # float("inf")

            self.sag_max = 8.0
            self.grad_max = 1.0
            self.grad2_max = 100.0

    # ================================================================
    # Loss functions
    # ================================================================
    def loss_reg(self, w_focus=None):
        """An empirical regularization loss for lens design. By default we should use weight 0.1 * self.loss_reg() in the total loss."""
        loss_focus = self.loss_infocus()

        if self.is_cellphone:
            loss_intersec = self.loss_self_intersec()
            loss_surf = self.loss_surface()
            loss_angle = self.loss_ray_angle()

            w_focus = 2.0 if w_focus is None else w_focus
            loss_reg = (
                w_focus * loss_focus
                + 1.0 * loss_intersec
                + 1.0 * loss_surf
                + 0.1 * loss_angle
            )
        else:
            loss_intersec = self.loss_self_intersec()
            loss_surf = self.loss_surface()
            loss_angle = self.loss_ray_angle()

            w_focus = 5.0 if w_focus is None else w_focus
            loss_reg = (
                w_focus * loss_focus
                + 1.0 * loss_intersec
                + 1.0 * loss_surf
                + 0.05 * loss_angle
            )

        return loss_reg

    def loss_infocus(self, bound=0.005):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            bound (float, optional): bound of RMS loss. Defaults to 0.005 [mm].
        """
        focz = self.d_sensor
        loss = []
        for wv in WAVE_RGB:
            # Ray tracing
            ray = self.sample_parallel(
                fov_x=0.0, fov_y=0.0, num_rays=SPP_CALC, wvln=wv, entrance_pupil=True
            )
            ray, _ = self.trace(ray)
            p = ray.project_to(focz)

            # Calculate RMS spot size as loss function
            rms_size = torch.sqrt(
                torch.sum((p**2 + EPSILON) * ray.ra.unsqueeze(-1))
                / (torch.sum(ray.ra) + EPSILON)
            )
            loss.append(max(rms_size, bound))

        loss_avg = sum(loss) / len(loss)
        return loss_avg

    def loss_rms(
        self,
        num_grid=GEO_GRID,
        depth=DEPTH,
        num_rays=SPP_CALC,
        importance_sampling=False,
    ):
        """Compute RGB RMS error per pixel, forward rms error.

        Can also revise this function to plot PSF.
        """
        # PSF and RMS by patch
        all_rms_errors = []
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            ray = self.sample_point_source(
                depth=depth,
                num_rays=num_rays,
                num_grid=num_grid,
                wvln=wvln,
                importance_sampling=importance_sampling,
            )
            ray = self.trace2sensor(ray)

            # Green light point center for reference
            if i == 0:
                pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(
                    -2
                ) / ray.ra.sum(-1).add(EPSILON).unsqueeze(
                    -1
                )  # shape [num_grid, num_grid, 2]
                pointc_green = pointc_green.unsqueeze(-2).repeat(
                    1, 1, num_rays, 1
                )  # shape [1, num_grid, num_grid, 2]

            # Calculate RMS error
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.ra.unsqueeze(-1)
            o2_norm = o2_norm[num_grid // 2, num_grid // 2, ...]
            ray.ra = ray.ra[num_grid // 2, num_grid // 2, ...]

            rms_error = torch.mean(
                (
                    ((o2_norm**2).sum(-1) * ray.ra).sum(-1) / (ray.ra.sum(-1) + EPSILON)
                ).sqrt()
            )
            all_rms_errors.append(rms_error)

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
                pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(
                    -2
                ) / ray.ra.sum(-1).add(EPSILON).unsqueeze(
                    -1
                )  # shape [1, num_fields, 2]
                pointc_green = pointc_green.unsqueeze(-2).repeat(
                    1, 1, SPP_PSF, 1
                )  # shape [num_fields, num_fields, num_rays, 2]

            # Calculate RMS error for different FoVs
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.ra.unsqueeze(-1)

            # error
            rms_error = torch.mean(
                (
                    ((o2_norm**2).sum(-1) * ray.ra).sum(-1) / (ray.ra.sum(-1) + EPSILON)
                ).sqrt()
            )

            # radius
            rms_radius = torch.mean(
                ((o2_norm**2).sum(-1) * ray.ra).sqrt().max(dim=-1).values
            )
            all_rms_errors.append(rms_error)
            all_rms_radii.append(rms_radius)

        # Calculate and print average across wavelengths
        avg_rms_error = torch.stack(all_rms_errors).mean(dim=0)
        avg_rms_radius = torch.stack(all_rms_radii).mean(dim=0)

        return avg_rms_error

    def loss_mtf(self, relative_fov=[0.0, 0.7, 1.0], depth=DEPTH, wvln=DEFAULT_WAVE):
        """Loss function designed on the MTF. We want to maximize MTF values."""
        loss = 0.0
        for fov in relative_fov:
            # ==> Calculate PSF
            point = torch.tensor([fov, fov, depth])
            psf = self.psf(points=point, wvln=wvln, ks=256)

            # ==> Calculate MTF
            x = torch.linspace(-1, 1, psf.shape[1]) * self.pixel_size * psf.shape[1] / 2
            y = torch.linspace(-1, 1, psf.shape[0]) * self.pixel_size * psf.shape[0] / 2

            # Extract 1D PSFs along the sagittal and tangential directions
            center_x = psf.shape[1] // 2
            center_y = psf.shape[0] // 2
            sagittal_psf = psf[center_y, :]
            tangential_psf = psf[:, center_x]

            # Fourier Transform to get the MTFs
            sagittal_mtf = torch.abs(torch.fft.fft(sagittal_psf))
            tangential_mtf = torch.abs(torch.fft.fft(tangential_psf))

            # Normalize the MTFs
            sagittal_mtf /= sagittal_mtf.max().detach()
            tangential_mtf /= tangential_mtf.max().detach()
            delta_x = self.pixel_size

            # Create frequency axis in cycles/mm
            freq = np.fft.fftfreq(psf.shape[0], delta_x)

            # Only keep the positive frequencies
            positive_freq_idx = freq > 0

            loss += torch.sum(
                sagittal_mtf[positive_freq_idx] + tangential_mtf[positive_freq_idx]
            ) / len(positive_freq_idx)

        return -loss

    def loss_fov(self, depth=DEPTH):
        """Trace rays from full FoV and converge them to the edge of the sensor. This loss term can constrain the FoV of the lens."""
        raise NotImplementedError("Need to check this function.")
        ray = self.sample_point_source_2D(depth=depth, num_rays=7, entrance_pupil=True)
        ray = self.trace2sensor(ray)
        loss = (
            (ray.o[:, 0] * ray.ra).sum() / (ray.ra.sum() + EPSILON) - self.r_sensor
        ).abs()
        return loss

    def loss_surface(self):
        """Penalize large sag, first-order derivative, and second-order derivative to prevent surface from being too curved."""
        sag_max = self.sag_max
        grad_max = self.grad_max
        grad2_max = self.grad2_max

        loss = 0.0
        for i in self.find_diff_surf():
            x_ls = torch.linspace(0.0, 1.0, 20).to(self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            loss += max(sag_ls.max() - sag_ls.min(), sag_max)

            # 1st-order derivative
            grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            loss += 10 * max(grad_ls.abs().max(), grad_max)

            # 2nd-order derivative
            grad2_ls = self.surfaces[i].d2fdxyz2(x_ls, y_ls)[0]
            loss += 10 * max(grad2_ls.abs().max(), grad2_max)

        return loss

    def loss_self_intersec(self):
        """Loss function to avoid self-intersection. Loss is designed by the distance to the next surfaces."""
        dist_min = self.dist_min
        dist_max = self.dist_max
        thickness_min = self.thickness_min
        thickness_max = self.thickness_max
        flange_min = self.flange_min
        flange_max = self.flange_max

        loss_min = 0.0
        loss_max = 0.0

        # Constraints for distance/thickness between surfaces
        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]

            r = torch.linspace(0.0, 1.0, 20).to(self.device) * current_surf.r
            z_front = current_surf.surface_with_offset(r, 0)
            z_next = next_surf.surface_with_offset(r, 0)

            # Minimum distance between surfaces
            dist_min = torch.min(z_next - z_front)
            if self.surfaces[i].mat2.name != "air":
                loss_min += min(thickness_min, dist_min)
            else:
                loss_min += min(dist_min, dist_min)

            # Maximum distance between surfaces
            dist_max = torch.max(z_next - z_front)
            if self.surfaces[i].mat2.name != "air":
                pass
            else:
                loss_max += max(thickness_max, dist_max)

        # Constraints for distance to the sensor (flange distance)
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 20).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0)
        loss_min += min(flange_min, torch.min(z_last_surf))
        loss_max += max(flange_max, torch.min(z_last_surf))

        return loss_max - loss_min

    def loss_ray_angle(self, target=0.5, depth=DEPTH):
        """Loss function designed to penalize large incident angle rays.

        Reference value: > 0.7
        """
        # Sample rays [512, M, M]
        M = GEO_GRID
        spp = 512
        ray = self.sample_point_source(depth=depth, num_rays=spp, num_grid=M)

        # Ray tracing
        ray, _ = self.trace(ray)

        # Loss (we want to maximize ray angle term)
        loss = ray.obliq.min()
        loss = min(loss, target)

        return -loss

    # ================================================================
    # Optimization script
    # ================================================================
    def optimize(
        self,
        lrs=[5e-4, 1e-4, 0.1, 1e-3],
        decay=0.01,
        iterations=2000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        match_mat=False,
        shape_control=True,
        importance_sampling=False,
        result_dir="./results",
    ):
        """Optimize the lens by minimizing rms errors.

        Debug hints:
            *, Slowly and continuously update!
            1, thickness (fov and ttl should match)
            2, alpha order (higher is better but more sensitive)
            3, learning rate and decay (prefer smaller lr and decay)
            4, correct params range
        """
        # Preparation
        depth = DEPTH
        num_grid = 31
        spp = 1024

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
        pbar = tqdm(total=iterations + 1, desc="Progress", postfix={"loss": 0})
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
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.sample_point_source(
                            depth=depth,
                            num_rays=spp,
                            num_grid=num_grid,
                            wvln=wv,
                            importance_sampling=importance_sampling,
                        )
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_p = -self.psf_center(
                            point=ray.o[:, :, 0, :], method="chief_ray"
                        )
                        center_p = center_p.unsqueeze(-2).repeat(1, 1, spp, 1)
                    else:
                        center_p = -self.psf_center(
                            point=ray.o[:, :, 0, :], method="pinhole"
                        )
                        center_p = center_p.unsqueeze(-2).repeat(1, 1, spp, 1)

            # ===> Optimize lens by minimizing RMS
            loss_rms = []
            for j, wv in enumerate(WAVE_RGB):
                # Ray tracing
                ray = rays_backup[j].clone()
                ray = self.trace2sensor(ray)
                xy = ray.o[..., :2]  # [h, w, spp, 2]
                ra = ray.ra.clone().detach()  # [h, w, spp]
                xy_norm = (xy - center_p) * ra.unsqueeze(-1)  # [h, w, spp, 2]

                # Use only quater of the sensor
                xy_norm = xy_norm[num_grid // 2 :, num_grid // 2 :, :, :]
                ra = ra[num_grid // 2 :, num_grid // 2 :, :]  # [h/2, w/2, spp]

                # Weight mask
                with torch.no_grad():
                    weight_mask = (xy_norm.clone().detach() ** 2).sum(-1).sqrt().sum(
                        -1
                    ) / (ra.sum(-1) + EPSILON)  # Use L2 error as weight mask
                    weight_mask /= weight_mask.mean()  # shape of [h/2, w/2]

                # Weighted L2 loss
                # l_rms = torch.mean(xy_norm.abs().sum(-1).sum(-1) / (ra.sum(-1) + EPSILON) * weight_mask)
                l_rms = torch.mean(
                    (xy_norm**2 + EPSILON).sum(-1).sqrt().sum(-1)
                    / (ra.sum(-1) + EPSILON)
                    * weight_mask
                )
                loss_rms.append(l_rms)

            loss_rms = sum(loss_rms) / len(loss_rms)

            # Total loss
            loss_reg = self.loss_reg()
            w_reg = 0.1
            L_total = loss_rms + w_reg * loss_reg

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=loss_rms.item())
            pbar.update(1)

        pbar.close()
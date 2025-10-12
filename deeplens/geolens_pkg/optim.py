# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Optimization and constraint functions for GeoLens.

Differentiable lens design has several advantages over conventional lens design:
    1. AutoDiff gradient calculation is faster and numerically more stable, which is important for complex optical systems.
    2. First-order optimization with momentum (e.g., Adam) is typically more stable than second-order optimization, and also has promising convergence speed.
    3. Efficient definition of loss functions can prevent the lens from violating constraints.

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
    """This class contains the optimization-related functions for the geometric lens design."""

    # ================================================================
    # Lens design constraints
    # ================================================================
    def init_constraints(self, constraint_params=None):
        """Initialize constraints for the lens design.
        
        Args:
            constraint_params (dict): Constraint parameters.
        """
        # In the future, we want to use constraint_params to set the constraints.
        if constraint_params is None:
            constraint_params = {}

        if self.r_sensor < 12.0:
            self.is_cellphone = True

            # Self intersection constraints
            self.dist_min_edge = 0.05
            self.dist_max_edge = 2.0
            self.dist_min_center = 0.05
            self.dist_max_center = 0.5
            
            self.thickness_min_edge = 0.25
            self.thickness_max_edge = 2.0
            self.thickness_min_center = 0.4
            self.thickness_max_center = 3.0
            
            self.flange_min = 0.8
            self.flange_max = 2.0

            # Surface shape constraints
            self.sag2diam_max = 0.1
            self.grad_max = 0.57 # tan(30deg)
            self.grad2_max = 50.0
            self.diam2thick_max = 10.0
            self.tmax2tmin_max = 5.0
            
            # Ray angle constraints
            self.chief_ray_angle_max = 30.0 # deg
            self.obliq_min = 0.6
        
        else:
            self.is_cellphone = False

            # Self-intersection constraints
            self.dist_min_edge = 0.1
            self.dist_max_edge = 50.0  # float("inf")
            self.dist_min_center = 0.1
            self.dist_max_center = 50.0  # float("inf")
            
            self.thickness_min_edge = 1.5
            self.thickness_max_edge = 10.0
            self.thickness_min_center = 2.5
            self.thickness_max_center = 15.0
            
            self.flange_min = 5.0
            self.flange_max = 50.0  # float("inf")

            # Surface shape constraints
            self.sag2diam_max = 0.1
            self.grad_max = 0.84 # tan(40deg)
            self.grad2_max = 100.0
            self.diam2thick_max = 10.0
            self.tmax2tmin_max = 5.0
            
            # Ray angle constraints
            self.chief_ray_angle_max = 20.0 # deg
            self.obliq_min = 0.4

    def loss_reg(self, w_focus=2.0, w_ray_angle=2.0, w_intersec=1.0, w_surf=1.0):
        """An empirical regularization loss for lens design.

        By default we should use weight 0.1 * self.loss_reg() in the total loss.
        """
        # Loss functions for regularization
        loss_focus = self.loss_infocus()
        loss_ray_angle = self.loss_ray_angle()
        loss_intersec = self.loss_self_intersec()
        loss_surf = self.loss_surface()
        # loss_mat = self.loss_mat()
        loss_reg = (
            w_focus * loss_focus
            + w_intersec * loss_intersec
            + w_surf * loss_surf
            + w_ray_angle * loss_ray_angle
            # + loss_mat
        )

        # Return loss and loss dictionary
        loss_dict = {
            "loss_focus": loss_focus.item(),
            "loss_intersec": loss_intersec.item(),
            "loss_surf": loss_surf.item(),
            'loss_ray_angle': loss_ray_angle.item(),
            # 'loss_mat': loss_mat.item(),
        }
        return loss_reg, loss_dict

    def loss_infocus(self, target=0.01):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            target (float, optional): target of RMS loss. Defaults to 0.005 [mm].
        """
        loss = torch.tensor(0.0, device=self.device)

        # Ray tracing and calculate RMS error
        ray = self.sample_parallel(fov_x=0.0, fov_y=0.0, wvln=WAVE_RGB[1], num_rays=SPP_CALC)
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
        sag2diam_max = self.sag2diam_max
        grad_max_allowed = self.grad_max
        grad2_max_allowed = self.grad2_max
        diam2thick_max = self.diam2thick_max
        tmax2tmin_max = self.tmax2tmin_max

        loss_grad = torch.tensor(0.0, device=self.device)
        loss_grad2 = torch.tensor(0.0, device=self.device)
        loss_diam2thick = torch.tensor(0.0, device=self.device)
        loss_tmax2tmin = torch.tensor(0.0, device=self.device)
        loss_sag2diam = torch.tensor(0.0, device=self.device)
        for i in self.find_diff_surf():
            # Sample points on the surface
            x_ls = torch.linspace(0.0, 1.0, 32).to(self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag2diam = sag_ls.abs().max() / self.surfaces[i].r / 2
            if sag2diam > sag2diam_max:
                loss_sag2diam += sag2diam

            # 1st-order derivative
            grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            grad_max = grad_ls.abs().max()
            if grad_max > grad_max_allowed:
                loss_grad += grad_max

            # 2nd-order derivative
            grad2_ls = self.surfaces[i].d2fdxyz2(x_ls, y_ls)[0]
            grad2_max = grad2_ls.abs().max()
            if grad2_max > grad2_max_allowed:
                loss_grad2 += 10 * grad2_max

            # Diameter to thickness ratio, thick_max to thick_min ratio
            if not self.surfaces[i].mat2.name == "air":
                surf2 = self.surfaces[i + 1]
                surf1 = self.surfaces[i]

                # Penalize diameter to thickness ratio
                d_to_t = max(surf2.r, surf1.r) / (surf2.d - surf1.d)
                if d_to_t > diam2thick_max:
                    loss_diam2thick += d_to_t

                # Penalize thick_max to thick_min ratio
                r_edge = min(surf2.r, surf1.r)
                thick_center = surf2.d - surf1.d
                thick_edge = surf2.surface_with_offset(r_edge, 0.0) - surf1.surface_with_offset(r_edge, 0.0)
                if thick_center > thick_edge:
                    tmax2tmin = thick_center / thick_edge
                else:
                    tmax2tmin = thick_edge / thick_center

                if tmax2tmin > tmax2tmin_max:
                    loss_tmax2tmin += tmax2tmin

        return loss_sag2diam + loss_grad + loss_grad2 + loss_diam2thick + loss_tmax2tmin

    def loss_self_intersec(self):
        """Loss function to avoid self-intersection.

        Loss is designed by the distance to the next surfaces.
        """
        # Constraints
        space_lower_center = self.dist_min_center
        space_upper_center = self.dist_max_center
        space_lower_edge = self.dist_min_edge
        space_upper_edge = self.dist_max_edge
        thick_lower_center = self.thickness_min_center
        thick_upper_center = self.thickness_max_center
        thick_lower_edge = self.thickness_min_edge
        thick_upper_edge = self.thickness_max_edge
        flange_min_allowed = self.flange_min
        flange_max_allowed = self.flange_max

        # Loss
        loss_min = torch.tensor(0.0, device=self.device)
        loss_max = torch.tensor(0.0, device=self.device)

        # Distance between surfaces
        for i in range(len(self.surfaces) - 1):
            # Sample evaluation points on the two surfaces
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]
            
            r_center = torch.tensor(0.0).to(self.device) * current_surf.r
            z_prev_center = current_surf.surface_with_offset(r_center, 0.0, valid_check=False)
            z_next_center = next_surf.surface_with_offset(r_center, 0.0, valid_check=False)
            
            r_edge = torch.linspace(0.5, 1.0, 16).to(self.device) * current_surf.r
            z_prev_edge = current_surf.surface_with_offset(r_edge, 0.0, valid_check=False)
            z_next_edge = next_surf.surface_with_offset(r_edge, 0.0, valid_check=False)

            # Air gap
            if self.surfaces[i].mat2.name == "air":
                # Center air gap
                dist_center = z_next_center - z_prev_center
                if dist_center < space_lower_center:
                    loss_min += dist_center

                if dist_center > space_upper_center:
                    loss_max += dist_center

                # Edge air gap
                dist_edge_min = torch.min(z_next_edge - z_prev_edge)
                if dist_edge_min < space_lower_edge:
                    loss_min += dist_edge_min
                
                dist_edge_max = torch.max(z_next_edge - z_prev_edge)
                if dist_edge_max > space_upper_edge:
                    loss_max += dist_edge_max

            # Lens thickness
            else:
                # Center thickness
                dist_center = z_next_center - z_prev_center
                if dist_center < thick_lower_center:
                    loss_min += dist_center

                if dist_center > thick_upper_center:
                    loss_max += dist_center

                # Edge thickness
                dist_edge_min = torch.min(z_next_edge - z_prev_edge)
                if dist_edge_min < thick_lower_edge:
                    loss_min += dist_edge_min

                dist_edge_max = torch.max(z_next_edge - z_prev_edge)
                if dist_edge_max > thick_upper_edge:
                    loss_max += dist_edge_max

        # Distance to sensor (flange)
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 32).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        
        flange = torch.min(z_last_surf)
        if flange < flange_min_allowed:
            loss_min += flange
        if flange > flange_max_allowed:
            loss_max += flange

        # Loss, minimize loss_max and maximize loss_min
        return loss_max - loss_min

    def loss_ray_angle(self):
        """Loss function to penalize large chief ray angle."""
        max_angle_deg = self.chief_ray_angle_max
        obliq_min = self.obliq_min

        # Loss on chief ray angle
        ray = self.sample_ring_arm_rays(num_ring=8, num_arm=8, spp=SPP_CALC, scale_pupil=0.2)
        ray = self.trace2sensor(ray)
        cos_cra = ray.d[..., 2]
        cos_cra_ref = float(np.cos(np.deg2rad(max_angle_deg)))
        if (cos_cra < cos_cra_ref).any():
            loss_cra = - cos_cra[cos_cra < cos_cra_ref].mean()
        else:
            loss_cra = torch.tensor(0.0, device=self.device)

        # Loss on accumulated oblique term
        ray = self.sample_ring_arm_rays(num_ring=8, num_arm=8, spp=SPP_CALC, scale_pupil=1.0)
        ray = self.trace2sensor(ray)
        obliq = ray.obliq.squeeze(-1)
        if (obliq < obliq_min).any():
            loss_obliq = - obliq[obliq < obliq_min].mean()
        else:
            loss_obliq = torch.tensor(0.0, device=self.device)

        return loss_cra + loss_obliq

    def loss_mat(self):
        n_max = 1.9
        n_min = 1.5
        V_max = 70
        V_min = 30
        loss_mat = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat1.name != "air":
                if self.surfaces[i].mat1.n > n_max:
                    loss_mat += (self.surfaces[i].mat1.n - n_max) / (n_max - n_min)
                if self.surfaces[i].mat1.n < n_min:
                    loss_mat += (n_min - self.surfaces[i].mat1.n) / (n_max - n_min)
                if self.surfaces[i].mat1.V > V_max:
                    loss_mat += (self.surfaces[i].mat1.V - V_max) / (V_max - V_min)
                if self.surfaces[i].mat1.V < V_min:
                    loss_mat += (V_min - self.surfaces[i].mat1.V) / (V_max - V_min)
        
        return loss_mat

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
        """Loss function to compute RGB spot error RMS.

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
                depth=depth,
                num_grid=num_grid,
                num_rays=num_rays,
                wvln=wvln,
                sample_more_off_axis=sample_more_off_axis,
            )

            # Calculate reference center, shape of (..., 2)
            if i == 0:
                with torch.no_grad():
                    ray_center_green = -self.psf_center(points=ray.o[:, :, 0, :], method="pinhole")

            ray = self.trace2sensor(ray)

            # # Green light centroid for reference
            # if i == 0:
            #     with torch.no_grad():
            #         ray_center_green = ray.centroid()

            # Calculate RMS error with reference center
            rms_error = ray.rms_error(center_ref=ray_center_green)
            all_rms_errors.append(rms_error)

        # Calculate average RMS error
        avg_rms_error = torch.stack(all_rms_errors).mean(dim=0)
        return avg_rms_error

    # ================================================================
    # Example optimization function
    # ================================================================
    def sample_ring_arm_rays(self, num_ring=8, num_arm=8, spp=2048, depth=DEPTH, wvln=DEFAULT_WAVE, scale_pupil=1.0, sample_more_off_axis=True):
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
        max_fov_rad = self.rfov
        if sample_more_off_axis:
            # Use beta distribution to sample more points near the edge (close to 1.0)
            # Beta(0.5, 0.5) gives more samples at 0 and 1, Beta(0.5, 0.3) gives more samples near 1.0
            beta_values = torch.linspace(0.0, 1.0, num_ring, device=self.device)
            # Apply beta transformation to concentrate samples near 1.0
            beta_transformed = beta_values ** 0.5  # Equivalent to Beta(0.5, 1.0) distribution
            ring_fovs = max_fov_rad * beta_transformed
        else:
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
            1, thickness (fov and ttl better match)
            2, alpha order (higher is better but more sensitive)
            3, learning rate and decay (prefer smaller lr and decay)
            4, reasonable parameters range
            5. nan can be introduced by torch.sqrt() function in the backward pass.
        """
        # Experiment settings
        depth = DEPTH
        num_ring = 32
        num_arm = 8
        spp = 2048

        # Result directory and logger
        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-DesignLens"

        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(f"lr:{lrs}, iterations:{iterations}, num_ring:{num_ring}, num_arm:{num_arm}, rays_per_fov:{spp}.")
        logging.info("If Out-of-Memory, try to reduce num_ring, num_arm, and rays_per_fov.")

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
                        # self.refocus()

                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}")
            
                    # Sample rays
                    self.calc_pupil()
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.sample_ring_arm_rays(num_ring=num_ring, num_arm=num_arm, spp=spp, depth=depth, wvln=wv, scale_pupil=1.05, sample_more_off_axis=False)
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_ref = -self.psf_center(points=ray.o[:, :, 0, :], method="chief_ray")
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)
                    else:
                        center_ref = -self.psf_center(points=ray.o[:, :, 0, :], method="pinhole")
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
                l_rms = (((ray_err**2).sum(-1) + EPSILON).sqrt() * ray_valid).sum(-1) # l2 loss
                # l_rms = (ray_err.abs().sum(-1) * ray_valid).sum(-1) # l1 loss
                l_rms /= ray_valid.sum(-1) + EPSILON

                # Weighted loss
                l_rms_weighted = (l_rms * weight_mask).sum()
                l_rms_weighted /= weight_mask.sum() + EPSILON
                loss_rms_ls.append(l_rms_weighted)

            # RMS loss for all wavelengths
            loss_rms = sum(loss_rms_ls) / len(loss_rms_ls)

            # Total loss
            loss_reg, loss_dict = self.loss_reg()
            w_reg = 0.1
            L_total = loss_rms + w_reg * loss_reg

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
            pbar.update(1)

        pbar.close()

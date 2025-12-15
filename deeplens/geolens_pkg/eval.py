# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Classical optical performance evaluation for geometric lens systems. Accuracy aligned with Zemax.

Functions:
    Spot Diagram:
        - draw_spot_radial(): Draw spot diagrams at different field angles along meridional direction
        - draw_spot_map(): Draw spot diagram grid at different field angles

    RMS Error:
        - rms_map_rgb(): Calculate RMS spot error map for RGB wavelengths
        - rms_map(): Calculate RMS spot error map for a specific wavelength

    Distortion:
        - calc_distortion_2D(): Calculate distortion at a specific field angle
        - draw_distortion_radial(): Draw distortion curve vs field angle (Zemax format)
        - distortion_map(): Compute distortion map at a given depth
        - draw_distortion(): Draw distortion map visualization

    MTF (Modulation Transfer Function):
        - mtf(): Calculate MTF at a specific field of view
        - psf2mtf(): Convert PSF to MTF (static method)
        - draw_mtf(): Draw grid of MTF curves for multiple depths/FOVs and RGB wavelengths

    Field Curvature:
        - draw_field_curvature(): Draw field curvature visualization

    Vignetting:
        - vignetting(): Compute vignetting map
        - draw_vignetting(): Draw vignetting visualization

    Wavefront & Aberration (placeholders):
        - wavefront_error(): Compute wavefront error
        - field_curvature(): Compute field curvature
        - aberration_histogram(): Compute aberration histogram

    Chief Ray & Ray Aiming:
        - calc_chief_ray(): Compute chief ray for an incident angle
        - calc_chief_ray_infinite(): Compute chief ray for infinite object distance
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deeplens.basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from deeplens.optics.ray import Ray

# RGB color definitions for wavelength visualization
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def draw_spot_radial(
        self,
        save_name="./lens_spot_radial.png",
        num_fov=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln_list=WAVE_RGB,
        show=False,
    ):
        """Draw spot diagram of the lens at different field angles along meridional (y) direction.

        Args:
            save_name (string, optional): filename to save. Defaults to "./lens_spot_radial.png".
            num_fov (int, optional): field of view number. Defaults to 4.
            depth (float, optional): depth of the point source. Defaults to float("inf").
            num_rays (int, optional): number of rays to sample. Defaults to SPP_PSF.
            wvln_list (list, optional): wavelength list to render.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        assert isinstance(wvln_list, list), "wvln_list must be a list"

        # Prepare figure
        fig, axs = plt.subplots(1, num_fov, figsize=(num_fov * 3.5, 3))
        axs = np.atleast_1d(axs)

        # Trace and draw each wavelength separately, overlaying results
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays along meridional (y) direction, shape [num_fov, num_rays, 3]
            ray = self.sample_radial_rays(
                num_field=num_fov, depth=depth, num_rays=num_rays, wvln=wvln
            )

            # Trace rays to sensor plane, shape [num_fov, num_rays, 3]
            ray = self.trace2sensor(ray)
            ray_o = ray.o.clone().cpu().numpy()
            ray_valid = ray.is_valid.clone().cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Plot multiple spot diagrams in one figure
            for i in range(num_fov):
                valid = ray_valid[i, :]
                x, y = ray_o[i, :, 0], ray_o[i, :, 1]

                # Filter valid rays
                mask = valid > 0
                x_valid, y_valid = x[mask], y[mask]

                # Plot points and center of mass for this wavelength
                axs[i].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                axs[i].set_aspect("equal", adjustable="datalim")
                axs[i].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def draw_spot_map(
        self,
        save_name="./lens_spot_map.png",
        num_grid=5,
        depth=DEPTH,
        num_rays=SPP_PSF,
        wvln_list=WAVE_RGB,
        show=False,
    ):
        """Draw spot diagram of the lens at different field angles.

        Args:
            save_name (string, optional): filename to save. Defaults to "./lens_spot_map.png".
            num_grid (int, optional): number of grid points. Defaults to 5.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            num_rays (int, optional): number of rays to sample. Defaults to SPP_PSF.
            wvln_list (list, optional): wavelength list to render. Defaults to WAVE_RGB.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        assert isinstance(wvln_list, list), "wvln_list must be a list"

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(
            num_grid, num_grid, figsize=(num_grid * 3, num_grid * 3)
        )
        axs = np.atleast_2d(axs)

        # Loop wavelengths and overlay scatters
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays per wavelength, shape [num_grid, num_grid, num_rays, 3]
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln
            )
            # Trace rays to sensor
            ray = self.trace2sensor(ray)

            # Convert to numpy, shape [num_grid, num_grid, num_rays, 3]
            ray_o = -ray.o.clone().cpu().numpy()
            ray_valid = ray.is_valid.clone().cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Draw per grid cell
            for i in range(num_grid):
                for j in range(num_grid):
                    valid = ray_valid[i, j, :]
                    x, y = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                    # Filter valid rays
                    mask = valid > 0
                    x_valid, y_valid = x[mask], y[mask]

                    # Plot points for this wavelength
                    axs[i, j].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                    axs[i, j].set_aspect("equal", adjustable="datalim")
                    axs[i, j].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # RMS map
    # ================================================================
    @torch.no_grad()
    def rms_map_rgb(self, num_grid=32, depth=DEPTH):
        """Calculate the RMS spot error map across RGB wavelengths. Reference to the centroid of green rays.

        Args:
            num_grid (int, optional): Number of grid points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.

        Returns:
            rms_map (torch.Tensor): RMS map for RGB channels. Shape [3, num_grid, num_grid].
        """
        all_rms_maps = []

        # Iterate G, R, B
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
            )

            ray = self.trace2sensor(ray)
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid

            # Calculate green centroid as reference, shape [num_grid, num_grid, 2]
            if i == 0:
                ray_xy_center_green = (ray_xy * ray_valid.unsqueeze(-1)).sum(
                    -2
                ) / ray_valid.sum(-1).add(EPSILON).unsqueeze(-1)

            # Calculate RMS relative to green centroid, shape [num_grid, num_grid]
            rms_map = torch.sqrt(
                (
                    ((ray_xy - ray_xy_center_green.unsqueeze(-2)) ** 2).sum(-1)
                    * ray_valid
                ).sum(-1)
                / (ray_valid.sum(-1) + EPSILON)
            )
            all_rms_maps.append(rms_map)

        # Stack the RMS maps for R, G, B channels, shape [3, num_grid, num_grid]
        rms_map_rgb = torch.stack(
            [all_rms_maps[1], all_rms_maps[0], all_rms_maps[2]], dim=0
        )

        return rms_map_rgb

    @torch.no_grad()
    def rms_map(self, num_grid=32, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Calculate the RMS spot error map for a specific wavelength.

        Currently this function is not used, but it can be used as the weight mask during optimization.

        Args:
            num_grid (int, optional): Resolution of the grid used for sampling fields/points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.
            wvln (float, optional): Wavelength of the ray. Defaults to DEFAULT_WAVE.

        Returns:
            rms_map (torch.Tensor): RMS map for the specified wavelength. Shape [num_grid, num_grid].
        """
        # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
        )
        ray = self.trace2sensor(ray)
        ray_xy = ray.o[..., :2]  # Shape [num_grid, num_grid, spp, 2]
        ray_valid = ray.is_valid  # Shape [num_grid, num_grid, spp]

        # Calculate centroid for each field point for this wavelength
        ray_xy_center = (ray_xy * ray_valid.unsqueeze(-1)).sum(-2) / ray_valid.sum(
            -1
        ).add(EPSILON).unsqueeze(-1)
        # Shape [num_grid, num_grid, 2]

        # Calculate RMS error relative to its own centroid, shape [num_grid, num_grid]
        rms_map = torch.sqrt(
            (((ray_xy - ray_xy_center.unsqueeze(-2)) ** 2).sum(-1) * ray_valid).sum(-1)
            / (ray_valid.sum(-1) + EPSILON)
        )

        return rms_map

    # ================================================================
    # Distortion
    # ================================================================
    def calc_distortion_2D(
        self, rfov, wvln=DEFAULT_WAVE, plane="meridional", ray_aiming=True
    ):
        """Calculate distortion at a specific field angle.

        Args:
            rfov (float): view angle (degree)
            wvln (float): wavelength
            plane (str): meridional or sagittal
            ray_aiming (bool): whether the chief ray through the center of the stop.

        Returns:
            distortion (float): distortion at the specific field angle
        """
        # Calculate ideal image height
        eff_foclen = self.foclen
        ideal_imgh = eff_foclen * np.tan(rfov * np.pi / 180)

        # Calculate chief ray
        chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
            rfov=rfov, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )
        ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)

        ray, _ = self.trace(ray)
        t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # Calculate actual image height
        if plane == "sagittal":
            actual_imgh = (ray.o[..., 0] + ray.d[..., 0] * t).abs()
        elif plane == "meridional":
            actual_imgh = (ray.o[..., 1] + ray.d[..., 1] * t).abs()
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Calculate distortion
        actual_imgh = actual_imgh.cpu().numpy()
        ideal_imgh = ideal_imgh.cpu().numpy()
        distortion = (actual_imgh - ideal_imgh) / ideal_imgh

        # Handle the case where ideal_imgh is 0 or very close to 0
        mask = abs(ideal_imgh) < EPSILON
        distortion[mask] = 0.0

        return distortion

    def draw_distortion_radial(
        self,
        rfov,
        save_name=None,
        num_points=GEO_GRID,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        ray_aiming=True,
        show=False,
    ):
        """Draw distortion. zemax format(default): ray_aiming = False.

        Note: this function is provided by a community contributor.

        Args:
            rfov: view angle (degrees)
            save_name: Save filename. Defaults to None.
            num_points: Number of points. Defaults to GEO_GRID.
            plane: Meridional or sagittal. Defaults to meridional.
            ray_aiming: Whether to use ray aiming. Defaults to False.
        """
        # Sample view angles
        rfov_samples = torch.linspace(0, rfov, num_points)
        distortions = []

        # Calculate distortion
        distortions = self.calc_distortion_2D(
            rfov=rfov_samples,
            wvln=wvln,
            plane=plane,
            ray_aiming=ray_aiming,
        )

        # Handle possible NaN values and convert to percentage
        values = [
            t.item() * 100 if not math.isnan(t.item()) else 0 for t in distortions
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{plane} Surface Distortion")

        # Draw distortion curve
        ax.plot(values, rfov_samples, linestyle="-", color="g", linewidth=1.5)

        # Draw reference line (vertical line)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)

        # Set grid
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1)

        # Dynamically adjust x-axis range
        value = max(abs(v) for v in values)
        margin = value * 0.2  # 20% margin
        x_min, x_max = -max(0.2, value + margin), max(0.2, value + margin)

        # Set ticks
        x_ticks = np.linspace(-value, value, 3)
        y_ticks = np.linspace(0, rfov, 3)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Format tick labels
        x_labels = [f"{x:.1f}%" for x in x_ticks]
        y_labels = [f"{y:.1f}" for y in y_ticks]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Set axis labels
        ax.set_xlabel("Distortion (%)")
        ax.set_ylabel("Field of View (degrees)")

        # Set axis range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, rfov)

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = f"./{plane}_distortion_inf.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def distortion_map(self, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Compute distortion map at a given depth.

        Args:
            num_grid (int): number of grid points.
            depth (float): depth of the point source.
            wvln (float): wavelength.

        Returns:
            distortion_grid (torch.Tensor): distortion map. shape (grid_size, grid_size, 2)
        """
        # Sample and trace rays, shape (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid, wvln=wvln, uniform_fov=False)
        ray = self.trace2sensor(ray)

        # Calculate centroid of the rays, shape (grid_size, grid_size, 2)
        ray_xy = ray.centroid()[..., :2]
        x_dist = -ray_xy[..., 0] / self.sensor_size[1] * 2
        y_dist = ray_xy[..., 1] / self.sensor_size[0] * 2
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    def draw_distortion(
        self, save_name=None, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE, show=False
    ):
        """Draw distortion map.

        Args:
            save_name (str, optional): filename to save. Defaults to None.
            num_grid (int, optional): number of grid points. Defaults to 16.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength. Defaults to DEFAULT_WAVE.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        # Ray tracing to calculate distortion map
        distortion_grid = self.distortion_map(num_grid=num_grid, depth=depth, wvln=wvln)
        x1 = distortion_grid[..., 0].cpu().numpy()
        y1 = distortion_grid[..., 1].cpu().numpy()

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title("Lens distortion")
        ax.scatter(x1, y1, s=2)
        ax.axis("scaled")
        ax.grid(True)

        # Add grid lines based on grid_size
        ax.set_xticks(np.linspace(-1, 1, num_grid))
        ax.set_yticks(np.linspace(-1, 1, num_grid))

        if show:
            plt.show()
        else:
            depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
            if save_name is None:
                save_name = f"./distortion_{depth_str}.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # MTF
    # ================================================================
    def mtf(self, fov, wvln=DEFAULT_WAVE):
        """Calculate MTF at a specific field of view."""
        point = [0, -fov / self.rfov, DEPTH]
        psf = self.psf(points=point, recenter=True, wvln=wvln)
        freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size=self.pixel_size)
        return freq, mtf_tan, mtf_sag

    @staticmethod
    def psf2mtf(psf, pixel_size):
        """Calculate MTF from PSF.

        Args:
            psf (tensor): 2D PSF tensor (e.g., ks x ks). Assumes standard orientation where the array's y-axis corresponds to the tangential/meridional direction and the x-axis to the sagittal direction.
            pixel_size (float): Pixel size in mm.

        Returns:
            freq (ndarray): Frequency axis (cycles/mm).
            tangential_mtf (ndarray): Tangential MTF.
            sagittal_mtf (ndarray): Sagittal MTF.

        Reference:
            [1] https://en.wikipedia.org/wiki/Optical_transfer_function
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/introduction-to-modulation-transfer-function/?srsltid=AfmBOoq09vVDVlh_uuwWnFoMTg18JVgh18lFSw8Ci4Sdlry-AmwGkfDd
        """
        # Convert to numpy (supports torch tensors and numpy arrays)
        try:
            psf_np = psf.detach().cpu().numpy()
        except AttributeError:
            try:
                psf_np = psf.cpu().numpy()
            except AttributeError:
                psf_np = np.asarray(psf)

        # Compute line spread functions (integrate PSF over orthogonal axes)
        # y-axis corresponds to tangential; x-axis corresponds to sagittal
        lsf_sagittal = psf_np.sum(axis=0)  # function of x
        lsf_tangential = psf_np.sum(axis=1)  # function of y

        # One-sided spectra (for real inputs)
        mtf_sag = np.abs(np.fft.rfft(lsf_sagittal))
        mtf_tan = np.abs(np.fft.rfft(lsf_tangential))

        # Normalize by DC to ensure MTF(0) == 1
        dc_sag = mtf_sag[0] if mtf_sag.size > 0 else 1.0
        dc_tan = mtf_tan[0] if mtf_tan.size > 0 else 1.0
        if dc_sag != 0:
            mtf_sag = mtf_sag / dc_sag
        if dc_tan != 0:
            mtf_tan = mtf_tan / dc_tan

        # Frequency axis in cycles/mm (one-sided)
        fx = np.fft.rfftfreq(lsf_sagittal.size, d=pixel_size)
        freq = fx
        positive_freq_idx = freq > 0

        return (
            freq[positive_freq_idx],
            mtf_tan[positive_freq_idx],
            mtf_sag[positive_freq_idx],
        )

    @torch.no_grad()
    def draw_mtf(
        self,
        save_name="./lens_mtf.png",
        relative_fov_list=[0.0, 0.7, 1.0],
        depth_list=[DEPTH],
        psf_ks=128,
        show=False,
    ):
        """Draw a grid of MTF curves.
        Each subplot in the grid corresponds to a specific (depth, FOV) combination.
        Each subplot displays MTF curves for R, G, B wavelengths.

        Args:
            relative_fov_list (list, optional): List of relative field of view values. Defaults to [0.0, 0.7, 1.0].
            depth_list (list, optional): List of depth values. Defaults to [DEPTH].
            save_name (str, optional): Filename to save the plot. Defaults to "./mtf_grid.png".
            psf_ks (int, optional): Kernel size for intermediate PSF calculation. Defaults to 256.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        pixel_size = self.pixel_size
        nyquist_freq = 0.5 / pixel_size
        num_fovs = len(relative_fov_list)
        if float("inf") in depth_list:
            depth_list = [DEPTH if x == float("inf") else x for x in depth_list]
        num_depths = len(depth_list)

        # Create figure and subplots (num_depths * num_fovs subplots)
        fig, axs = plt.subplots(
            num_depths, num_fovs, figsize=(num_fovs * 3, num_depths * 3), squeeze=False
        )

        # Iterate over depth and field of view
        for depth_idx, depth in enumerate(depth_list):
            for fov_idx, fov_relative in enumerate(relative_fov_list):
                # Calculate rgb PSF
                point = [0, -fov_relative, depth]
                psf_rgb = self.psf_rgb(points=point, ks=psf_ks, recenter=False)

                # Calculate MTF curves for rgb wavelengths
                for wvln_idx, wvln in enumerate(WAVE_RGB):
                    # Calculate MTF curves from PSF
                    psf = psf_rgb[wvln_idx]
                    freq, mtf_tan, _ = self.psf2mtf(psf, pixel_size)

                    # Plot MTF curves
                    ax = axs[depth_idx, fov_idx]
                    color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]
                    wvln_label = RGB_LABELS[wvln_idx % len(RGB_LABELS)]
                    wvln_nm = int(wvln * 1000)
                    ax.plot(
                        freq,
                        mtf_tan,
                        color=color,
                        label=f"{wvln_label}({wvln_nm}nm)-Tan",
                    )

                # Draw Nyquist frequency
                ax.axvline(
                    x=nyquist_freq,
                    color="k",
                    linestyle=":",
                    linewidth=1.2,
                    label="Nyquist",
                )

                # Set title and label for subplot
                fov_deg = round(fov_relative * self.rfov * 180 / np.pi, 1)
                depth_str = "inf" if depth == float("inf") else f"{depth}"
                ax.set_title(f"FOV: {fov_deg}deg, Depth: {depth_str}mm", fontsize=8)
                ax.set_xlabel("Spatial Frequency [cycles/mm]", fontsize=8)
                ax.set_ylabel("MTF", fontsize=8)
                ax.legend(fontsize=6)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True)
                ax.set_ylim(0, 1.05)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Field Curvature
    # ================================================================
    @torch.no_grad()
    def draw_field_curvature(
        self,
        save_name=None,
        num_points=32,
        z_span=1.0,
        z_steps=1001,
        wvln_list=WAVE_RGB,
        spp=SPP_CALC,
        show=False,
    ):
        """Draw field curvature: best-focus defocus Δz (mm) vs field angle (deg), RGB overlaid.

        - Tangential (meridional) curves are solid lines (y-axis spread minimized).
        """
        print("This function is not optimized for the best speed.")
        device = self.device
        # Convert maximum field angle to degrees
        rfov_deg = float(self.rfov) * 180.0 / np.pi

        # Sample field angles [0, rfov_deg]
        rfov_samples = torch.linspace(0.0, rfov_deg, num_points, device=device)

        # Prepare containers
        delta_z_tan = []  # list of numpy arrays per wavelength

        # Defocus sweep grid (around current sensor plane)
        d_sensor = self.d_sensor
        z_grid = d_sensor + torch.linspace(-z_span, z_span, z_steps, device=device)

        # Helper to compute best focus along a given axis (0=x sagittal, 1=y tangential)
        def best_focus_delta_z(ray, axis_idx: int):
            # ray: after lens surfaces (image space)
            # Vectorized intersection with planes z_grid
            oz = ray.o[..., 2:3]
            dz = ray.d[..., 2:3]
            t = (z_grid.unsqueeze(0) - oz) / (dz + 1e-12)  # [N, Z]

            oa = ray.o[..., axis_idx : axis_idx + 1]
            da = ray.d[..., axis_idx : axis_idx + 1]
            pos_axis = (oa + da * t).squeeze(-1)  # [N, Z]

            w = ray.is_valid.unsqueeze(-1).float()  # [N, 1] -> [N, Z] by broadcast
            pos_axis = pos_axis * w
            w_sum = w.sum(0)  # [Z]
            centroid = pos_axis.sum(0) / (w_sum + EPSILON)  # [Z]
            ms = (((pos_axis - centroid.unsqueeze(0)) ** 2) * w).sum(0) / (
                w_sum + EPSILON
            )  # [Z]
            best_idx = torch.argmin(ms)
            return (z_grid[best_idx] - d_sensor).item()

        # Loop wavelengths and field angles
        for w_idx, wvln in enumerate(wvln_list):
            dz_tan = []
            for i in range(len(rfov_samples)):
                fov_deg = rfov_samples[i].item()

                # Tangential (meridional plane: y-z plane -> minimize y spread)
                ray_t = self.sample_parallel_2D(
                    fov=fov_deg,
                    num_rays=spp,
                    wvln=wvln,
                    plane="meridional",
                    entrance_pupil=True,
                )
                ray_t, _ = self.trace(ray_t)
                dz_tan.append(best_focus_delta_z(ray_t, axis_idx=1))  # y-axis

            delta_z_tan.append(np.asarray(dz_tan))

        # Plot
        fov_np = rfov_samples.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Field Curvature (Δz vs Field Angle)")

        # Determine x range (tangential only)
        all_vals = np.abs(np.concatenate(delta_z_tan)) if len(delta_z_tan) > 0 else np.array([0.0])
        x_range = float(max(0.2, all_vals.max() * 1.2)) if all_vals.size > 0 else 0.2

        for w_idx in range(len(wvln_list)):
            color = RGB_COLORS[w_idx % len(RGB_COLORS)]
            lbl = RGB_LABELS[w_idx % len(RGB_LABELS)]
            ax.plot(
                delta_z_tan[w_idx],
                fov_np,
                color=color,
                linestyle="-",
                label=f"{lbl}-Tan",
            )

        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1.0)
        ax.set_xlabel("Defocus Δz (mm) relative to sensor plane")
        ax.set_ylabel("Field Angle (deg)")
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(0, rfov_deg)
        ax.legend(fontsize=8)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = "./field_curvature.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Vignetting
    # ================================================================
    def vignetting(self, depth=DEPTH, num_grid=64):
        """Compute vignetting."""
        # Sample rays, shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid)

        # Trace rays to sensor
        ray = self.trace2sensor(ray)

        # Calculate vignetting map
        vignetting = ray.is_valid.sum(-1) / (ray.is_valid.shape[-1])
        return vignetting

    def draw_vignetting(self, filename=None, depth=DEPTH, resolution=512, show=False):
        """Draw vignetting."""
        # Calculate vignetting map
        vignetting = self.vignetting(depth=depth)

        # Interpolate vignetting map to desired resolution
        vignetting = F.interpolate(
            vignetting.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Scale vignetting to [0.5, 1] range
        vignetting = 0.5 + 0.5 * vignetting

        fig, ax = plt.subplots()
        ax.imshow(vignetting.cpu().numpy(), cmap="gray", vmin=0.5, vmax=1.0)
        ax.colorbar(ticks=[0.5, 0.75, 1.0])

        if show:
            plt.show()
        else:
            if filename is None:
                filename = f"./vignetting_{depth}.png"
            plt.savefig(filename, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Wavefront error
    # ================================================================
    def wavefront_error(self):
        """Compute wavefront error."""
        pass

    def field_curvature(self):
        """Compute field curvature."""
        pass

    def aberration_histogram(self):
        """Compute aberration histogram."""
        pass

    # ================================================================
    # Chief ray calculation and ray aiming
    # ================================================================
    @torch.no_grad()
    def calc_chief_ray(self, fov, plane="sagittal"):
        """Compute chief ray for an incident angle.

        If chief ray is only used to determine the ideal image height, we can warp this function into the image height calculation function.

        Args:
            fov (float): incident angle in degree.
            plane (str): "sagittal" or "meridional".

        Returns:
            chief_ray_o (torch.Tensor): origin of chief ray.
            chief_ray_d (torch.Tensor): direction of chief ray.

        Note:
            It is 2D ray tracing, for 3D chief ray, we can shrink the pupil, trace rays, calculate the centroid as the chief ray.
        """
        # Sample parallel rays from object space
        ray = self.sample_parallel_2D(
            fov=fov, num_rays=SPP_CALC, entrance_pupil=True, plane=plane
        )
        inc_ray = ray.clone()

        # Trace to the aperture
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Look for the ray that is closest to the optical axis
        center_x = torch.min(torch.abs(ray.o[:, 0]))
        center_idx = torch.where(torch.abs(ray.o[:, 0]) == center_x)[0][0].item()
        chief_ray_o, chief_ray_d = inc_ray.o[center_idx, :], inc_ray.d[center_idx, :]

        return chief_ray_o, chief_ray_d

    @torch.no_grad()
    def calc_chief_ray_infinite(
        self,
        rfov,
        depth=0.0,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        num_rays=SPP_CALC,
        ray_aiming=True,
    ):
        """Compute chief ray for an incident angle.

        Args:
            rfov (float): incident angle in degree.
            depth (float): depth of the object.
            wvln (float): wavelength of the light.
            plane (str): "sagittal" or "meridional".
            num_rays (int): number of rays.
            ray_aiming (bool): whether the chief ray through the center of the stop.
        """
        if isinstance(rfov, float) and rfov > 0:
            rfov = torch.linspace(0, rfov, 2)
        rfov = rfov.to(self.device)

        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=self.device).repeat(len(rfov))

        # set chief ray
        chief_ray_o = torch.zeros([len(rfov), 3]).to(self.device)
        chief_ray_d = torch.zeros([len(rfov), 3]).to(self.device)

        # Convert rfov to radian
        rfov = rfov * torch.pi / 180.0

        if torch.any(rfov == 0):
            chief_ray_o[0, ...] = torch.tensor(
                [0.0, 0.0, depth[0]], device=self.device, dtype=torch.float32
            )
            chief_ray_d[0, ...] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
            if len(rfov) == 1:
                return chief_ray_o, chief_ray_d

        if len(rfov) > 1:
            rfovs = rfov[1:]
            depths = depth[1:]

        if self.aper_idx == 0:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [depths * torch.tan(rfovs), torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), depths * torch.tan(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

            return chief_ray_o, chief_ray_d

        # Scale factor
        pupilz, _ = self.calc_entrance_pupil()
        y_distance = torch.tan(rfovs) * (abs(depths) + pupilz)

        if ray_aiming:
            scale = 0.05
            delta = scale * y_distance

        if not ray_aiming:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [-y_distance, torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), -y_distance, depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        else:
            min_y = -y_distance - delta
            max_y = -y_distance + delta
            o1_linspace = torch.stack(
                [
                    torch.linspace(min_y[i], max_y[i], num_rays)
                    for i in range(len(min_y))
                ],
                dim=0,
            )

            o1 = torch.zeros([len(rfovs), num_rays, 3])
            o1[:, :, 2] = depths[0]

            o2_linspace = torch.stack(
                [
                    torch.linspace(-delta[i], delta[i], num_rays)
                    for i in range(len(min_y))
                ],
                dim=0,
            )

            o2 = torch.zeros([len(rfovs), num_rays, 3])
            o2[:, :, 2] = pupilz

            if plane == "sagittal":
                o1[:, :, 0] = o1_linspace
                o2[:, :, 0] = o2_linspace
            else:
                o1[:, :, 1] = o1_linspace
                o2[:, :, 1] = o2_linspace

            # Trace until the aperture
            ray = Ray(o1, o2 - o1, wvln=wvln, device=self.device)
            inc_ray = ray.clone()
            surf_range = range(0, self.aper_idx + 1)
            ray, _ = self.trace(ray, surf_range=surf_range)

            # Look for the ray that is closest to the optical axis
            if plane == "sagittal":
                _, center_idx = torch.min(torch.abs(ray.o[..., 0]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                _, center_idx = torch.min(torch.abs(ray.o[..., 1]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        return chief_ray_o, chief_ray_d

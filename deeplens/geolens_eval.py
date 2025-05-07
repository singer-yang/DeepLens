"""Optical performance evaluation and visualization for GeoLens."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from deeplens.optics.basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from deeplens.optics.ray import Ray


class GeoLensEval:
    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def draw_spot_radial(
        self, num_field=5, depth=float("inf"), wvln=DEFAULT_WAVE, save_name=None
    ):
        """Draw spot diagram of the lens at different field angles along meridional (y) direction.

        Args:
            num_field (int, optional): field number. Defaults to 4.
            depth (float, optional): depth of the point source. Defaults to float("inf").
            wvln (float, optional): wavelength of the ray. Defaults to DEFAULT_WAVE.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample rays along meridional (y) direction, shape [num_field, num_rays, 3]
        ray = self.sample_radial_rays(
            num_field=num_field, depth=depth, num_rays=SPP_PSF, wvln=wvln
        )

        # Trace rays to sensor plane, shape [num_field, num_rays, 3]
        ray = self.trace2sensor(ray)
        ray_o = ray.o.clone().cpu().numpy() #.squeeze(0)
        ray_ra = ray.ra.clone().cpu().numpy() #.squeeze(0)

        # Plot multiple spot diagrams in one figure
        _, axs = plt.subplots(1, num_field, figsize=(num_field * 4, 4))
        for i in range(num_field):
            ra = ray_ra[i, :]
            x, y = ray_o[i, :, 0], ray_o[i, :, 1]

            # Filter valid rays
            x_valid, y_valid = x[ra > 0], y[ra > 0]
            ra_valid = ra[ra > 0]

            # Calculate center of mass for valid rays
            if ra_valid.sum() > EPSILON:
                xc, yc = x_valid.sum() / ra_valid.sum(), y_valid.sum() / ra_valid.sum()
            else:
                xc, yc = 0.0, 0.0

            # Plot points and center of mass
            axs[i].scatter(x_valid, y_valid, 3, "black", alpha=0.5)
            axs[i].scatter([xc], [yc], 100, "r", "x")
            axs[i].set_aspect("equal", adjustable="datalim")
            axs[i].tick_params(axis="both", which="major", labelsize=6)

        # Save plot
        depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
        if save_name is None:
            save_name = f"./spot_meridional_{depth_str}.png"
        else:
            if save_name.endswith(".png"):
                save_name = save_name[:-4]
            save_name = f"{save_name}_meridional_{depth_str}.png"

        plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close()

    @torch.no_grad()
    def draw_spot_map(self, num_grid=5, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """Draw spot diagram of the lens at different field angles.

        Args:
            num_grid (int, optional): number of grid points. Defaults to 5.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength of the ray. Defaults to DEFAULT_WAVE.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample rays, shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
        )

        # Trace rays to sensor
        ray = self.trace2sensor(ray)

        # Convert to numpy, shape [num_grid, num_grid, num_rays, 3]
        ray_o = -ray.o.clone().cpu().numpy()
        ray_ra = ray.ra.clone().cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(
            num_grid, num_grid, figsize=(num_grid * 2, num_grid * 2)
        )
        for i in range(num_grid):
            for j in range(num_grid):
                ra = ray_ra[i, j, :]
                x, y = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                # Filter valid rays
                x_valid, y_valid = x[ra > 0], y[ra > 0]
                ra_valid = ra[ra > 0]

                # Calculate center of mass for valid rays
                if ra_valid.sum() > EPSILON:
                    xc, yc = (
                        x_valid.sum() / ra_valid.sum(),
                        y_valid.sum() / ra_valid.sum(),
                    )
                else:
                    xc, yc = 0.0, 0.0

                # Plot points and center of mass
                axs[i, j].scatter(x_valid, y_valid, 2, "black", alpha=0.5)
                axs[i, j].scatter([xc], [yc], 100, "r", "x")
                axs[i, j].set_aspect("equal", adjustable="datalim")
                axs[i, j].tick_params(axis="both", which="major", labelsize=6)

        # Save plot
        depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
        if save_name is None:
            save_name = f"./spot_{depth_str}.png"
        else:
            if save_name.endswith(".png"):
                save_name = save_name[:-4]
            save_name = f"{save_name}_spot_{depth_str}.png"

        plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close()

    # ================================================================
    # RMS map
    # ================================================================
    @torch.no_grad()
    def rms_map_rgb(self, num_grid=64, depth=DEPTH):
        """Calculate the RMS spot error map across RGB wavelengths. Reference to the centroid of green rays.

        Args:
            num_grid (int, optional): Number of grid points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.

        Returns:
            rms_map (torch.Tensor): RMS map for RGB channels. Shape [3, num_grid, num_grid].
        """
        all_rms_maps = []

        # Iterate G, R, B
        for i, wvln_current in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
            ray = self.sample_grid_source(
                depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln_current
            )
            ray = self.trace2sensor(ray)
            ray_xy = ray.o[..., :2]
            ray_ra = ray.ra

            # Calculate green centroid as reference, shape [num_grid, num_grid, 2]
            if i == 0:
                ray_xy_center_green = (ray_xy * ray_ra.unsqueeze(-1)).sum(
                    -2
                ) / ray_ra.sum(-1).add(EPSILON).unsqueeze(-1)

            # Calculate RMS relative to green centroid, shape [num_grid, num_grid]
            rms_map = torch.sqrt(
                (
                    ((ray_xy - ray_xy_center_green.unsqueeze(-2)) ** 2).sum(-1) * ray_ra
                ).sum(-1)
                / (ray_ra.sum(-1) + EPSILON)
            )
            all_rms_maps.append(rms_map)

        # Stack the RMS maps for R, G, B channels, shape [3, num_grid, num_grid]
        rms_map_rgb = torch.stack(
            [all_rms_maps[1], all_rms_maps[0], all_rms_maps[2]], dim=0
        )

        return rms_map_rgb

    @torch.no_grad()
    def rms_map(self, num_grid=64, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Calculate the RMS spot error map for a specific wavelength.

        Args:
            num_grid (int, optional): Resolution of the grid used for sampling fields/points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.
            wvln (float, optional): Wavelength of the ray. Defaults to DEFAULT_WAVE.

        Returns:
            rms_map (torch.Tensor): RMS map for the specified wavelength. Shape [num_grid, num_grid].
        """
        # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
        ray = self.sample_grid_source(
            depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
        )
        ray = self.trace2sensor(ray)
        ray_xy = ray.o[..., :2]  # Shape [num_grid, num_grid, spp, 2]
        ray_ra = ray.ra  # Shape [num_grid, num_grid, spp]

        # Calculate centroid for each field point for this wavelength
        ray_xy_center = (ray_xy * ray_ra.unsqueeze(-1)).sum(-2) / ray_ra.sum(-1).add(
            EPSILON
        ).unsqueeze(-1)
        # Shape [num_grid, num_grid, 2]

        # Calculate RMS error relative to its own centroid, shape [num_grid, num_grid]
        rms_map = torch.sqrt(
            (((ray_xy - ray_xy_center.unsqueeze(-2)) ** 2).sum(-1) * ray_ra).sum(
                -1
            )
            / (ray_ra.sum(-1) + EPSILON)
        )

        return rms_map

    # ================================================================
    # Distortion
    # ================================================================
    def calc_distortion_2D(
        self, hfov, wvln=DEFAULT_WAVE, plane="meridional", ray_aiming=True
    ):
        """Calculate distortion at a specific field angle.

        Args:
            hfov (float): view angle (degree)
            wvln (float): wavelength
            plane (str): meridional or sagittal
            ray_aiming (bool): whether the chief ray through the center of the stop.

        Returns:
            distortion (float): distortion at the specific field angle
        """
        # ===========>
        # # Calculate ideal image height
        # eff_foclen = self.calc_efl()
        # ideal_imgh = eff_foclen * np.tan(hfov * np.pi / 180)

        # # Calculate chief ray
        # chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
        #     hfov=hfov, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        # )
        # ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)

        # ray, _ = self.trace(ray, lens_range=range(len(self.surfaces)))
        # t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # # Calculate actual image height
        # if plane == "sagittal":
        #     actual_imgh = (ray.o[..., 0] + ray.d[..., 0] * t).abs()
        # elif plane == "meridional":
        #     actual_imgh = (ray.o[..., 1] + ray.d[..., 1] * t).abs()
        # else:
        #     raise ValueError(f"Invalid plane: {plane}")

        # # Calculate distortion
        # actual_imgh = actual_imgh.cpu().numpy()
        # ideal_imgh = ideal_imgh.cpu().numpy()
        # distortion = (actual_imgh - ideal_imgh) / ideal_imgh

        # # Handle the case where ideal_imgh is 0 or very close to 0
        # mask = abs(ideal_imgh) < EPSILON
        # distortion[mask] = 0.0
        # ===========>



        # ===========>

        return distortion

    def draw_distortion_2D(
        self,
        hfov,
        filename=None,
        num_points=GEO_GRID,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        ray_aiming=True,
    ):
        """Draw distortion. zemax format(default): ray_aiming = False.

        Args:
            hfov: view angle (degrees)
            filename: Save filename. Defaults to None.
            num_points: Number of points. Defaults to GEO_GRID.
            plane: Meridional or sagittal. Defaults to meridional.
            ray_aiming: Whether to use ray aiming. Defaults to False.
        """

        # Sample view angles
        hfov_samples = torch.linspace(0, hfov, num_points)
        distortions = []

        # Calculate distortion
        distortions = self.calc_distortion_2D(
            hfov=hfov_samples,
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
        ax.plot(values, hfov_samples, linestyle="-", color="g", linewidth=1.5)

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
        y_ticks = np.linspace(0, hfov, 3)

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
        ax.set_ylim(0, hfov)
        if filename is None:
            plt.savefig(
                f"./{plane}_distortion_infinite_mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )
        else:
            plt.savefig(
                f"{filename[:-4]}_{plane}_distortion_infinite_mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )

    def distortion(self, depth=DEPTH, grid_size=64):
        """Compute distortion map at a given depth.

        Args:
            depth (float): depth of the point source.
            img_res (tuple): resolution of the image.

        Returns:
            distortion_grid (torch.Tensor): distortion map. shape (grid_size, grid_size, 2)
        """
        # Ray tracing to calculate distortion map
        ray = self.sample_point_source(
            depth=depth, num_rays=SPP_CALC, num_grid=grid_size
        )
        ray = self.trace2sensor(ray)
        o_dist = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(-2) / ray.ra.unsqueeze(
            -1
        ).sum(-2).add(EPSILON)  # shape (H, W, 2)

        x_dist = -o_dist[..., 0] / self.sensor_size[1] * 2
        y_dist = o_dist[..., 1] / self.sensor_size[0] * 2
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)  # shape (H, W, 2)
        return distortion_grid

    def draw_distortion(self, filename=None, depth=DEPTH, grid_size=16):
        """Draw distortion."""
        # Ray tracing to calculate distortion map
        distortion_grid = self.distortion(depth=depth, grid_size=grid_size)
        x1 = distortion_grid[..., 0].cpu().numpy()
        y1 = distortion_grid[..., 1].cpu().numpy()

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title("Lens distortion")
        ax.scatter(x1, y1, s=2)
        ax.axis("scaled")
        ax.grid(True)

        # Add grid lines based on grid_size
        ax.set_xticks(np.linspace(-1, 1, grid_size))
        ax.set_yticks(np.linspace(-1, 1, grid_size))

        if filename is None:
            plt.savefig(
                f"./distortion{-depth}mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )
        else:
            plt.savefig(
                f"{filename[:-4]}_distortion_{-depth}mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )

    # ================================================================
    # MTF
    # ================================================================
    @torch.no_grad()
    def draw_mtf(
        self,
        wvlns=DEFAULT_WAVE,
        depth=DEPTH,
        relative_fov=[0.0, 0.7, 1.0],
        save_name="./mtf.png",
    ):
        """Draw MTF curve (different FoVs, single wvln, infinite depth) of the lens."""
        assert save_name[-4:] == ".png", "save_name must end with .png"

        relative_fov = (
            [relative_fov] if isinstance(relative_fov, float) else relative_fov
        )
        wvlns = [wvlns] if isinstance(wvlns, float) else wvlns
        color_list = "rgb"

        plt.figure(figsize=(6, 6))
        for wvln_idx, wvln in enumerate(wvlns):
            for fov_idx, fov in enumerate(relative_fov):
                point = torch.tensor([0, fov, depth])
                psf = self.psf(points=point, wvln=wvln, ks=256)
                freq, mtf_tan, mtf_sag = self.psf2mtf(psf)

                fov_deg = round(fov * self.hfov * 180 / torch.pi, 1)
                plt.plot(
                    freq,
                    mtf_tan,
                    color_list[fov_idx],
                    label=f"{fov_deg}(deg)-Tangential",
                )
                plt.plot(
                    freq,
                    mtf_sag,
                    color_list[fov_idx],
                    label=f"{fov_deg}(deg)-Sagittal",
                    linestyle="--",
                )

        plt.legend()
        plt.xlabel("Spatial Frequency [cycles/mm]")
        plt.ylabel("MTF")

        # Save figure
        plt.savefig(f"{save_name}", bbox_inches="tight", format="png", dpi=300)
        plt.close()

    def psf2mtf(self, psf, diag=False):
        """Convert 2D PSF kernel to MTF curve by FFT.

        Args:
            psf (tensor): 2D PSF tensor.

        Returns:
            freq (ndarray): Frequency axis.
            tangential_mtf (ndarray): Tangential MTF.
            sagittal_mtf (ndarray): Sagittal MTF.
        """
        psf = psf.cpu().numpy()
        x = np.linspace(-1, 1, psf.shape[1]) * self.pixel_size * psf.shape[1] / 2
        y = np.linspace(-1, 1, psf.shape[0]) * self.pixel_size * psf.shape[0] / 2

        if diag:
            raise Exception("Diagonal PSF is not tested.")
            diag_psf = np.diag(np.flip(psf, axis=0))
            x *= math.sqrt(2)
            y *= math.sqrt(2)
            delta_x = self.pixel_size * math.sqrt(2)

            diag_mtf = np.abs(np.fft.fft(diag_psf))
            # diag_mtf /= diag_mtf.max()

            # Create frequency axis in cycles/mm
            freq = np.fft.fftfreq(psf.shape[0], delta_x)

            # Only keep the positive frequencies
            positive_freq_idx = freq > 0

            freq = freq[positive_freq_idx]
            diag_mtf = diag_mtf[positive_freq_idx]
            diag_mtf /= diag_mtf[0]

            return freq, diag_mtf
        else:
            # Extract 1D PSFs along the sagittal and tangential directions
            center_x = psf.shape[1] // 2
            center_y = psf.shape[0] // 2
            sagittal_psf = psf[center_y, :]
            tangential_psf = psf[:, center_x]

            # Fourier Transform to get the MTFs
            sagittal_mtf = np.abs(np.fft.fft(sagittal_psf))
            tangential_mtf = np.abs(np.fft.fft(tangential_psf))

            # Normalize the MTFs
            sagittal_mtf /= sagittal_mtf.max()
            tangential_mtf /= tangential_mtf.max()

            delta_x = self.pixel_size  # / 2

            # Create frequency axis in cycles/mm
            freq = np.fft.fftfreq(psf.shape[0], delta_x)

            # Only keep the positive frequencies
            positive_freq_idx = freq > 0

            return (
                freq[positive_freq_idx],
                tangential_mtf[positive_freq_idx],
                sagittal_mtf[positive_freq_idx],
            )

    # ================================================================
    # Vignetting
    # ================================================================
    def vignetting(self):
        """Compute vignetting."""
        pass

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
    # 3D layout
    # ================================================================
    def draw_layout_3d(self, filename=None, figsize=(10, 6), view_angle=30, show=True):
        """Draw 3D layout of the lens system.

        Args:
            filename (str, optional): Path to save the figure. Defaults to None.
            figsize (tuple): Figure size
            view_angle (int): Viewing angle for the 3D plot
            show (bool): Whether to display the figure

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        from deeplens.geolens_utils import draw_layout_3d

        return draw_layout_3d(
            self, filename=filename, figsize=figsize, view_angle=view_angle, show=show
        )

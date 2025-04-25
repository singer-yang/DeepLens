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
    PSF_KS,
    SPP_CALC,
    SPP_PSF,
)
from deeplens.optics.ray import Ray


class GeoLensEval:
    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def draw_spot_radial(self, num_fields=4, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """Draw spot diagram of the lens at different fields along meridional direction.

        Args:
            num_fields (int, optional): field number. Defaults to 4.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength of the ray. Defaults to DEFAULT_WAVE.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample and trace rays, shape [num_fields, num_fields, num_rays, 3]
        ray = self.sample_point_source(
            depth=depth,
            num_rays=SPP_PSF,
            num_grid=[num_fields * 2 - 1, num_fields * 2 - 1],
            wvln=wvln,
        )
        ray = self.trace2sensor(ray)
        ray_o = torch.flip(ray.o.clone(), [0, 1]).cpu().numpy()
        ray_ra = torch.flip(ray.ra.clone(), [0, 1]).cpu().numpy()

        # Plot multiple spot diagrams in one figure
        _, axs = plt.subplots(1, num_fields, figsize=(num_fields * 4, 4))
        center_idx = num_fields - 1  # Index corresponding to the center of the grid
        for i in range(num_fields):
            # Select spots along the y-axis (meridional direction) starting from the center
            row_idx = center_idx - i
            col_idx = center_idx

            # Calculate center of mass
            ra = ray_ra[row_idx, col_idx, :]
            x, y = ray_o[row_idx, col_idx, :, 0], ray_o[row_idx, col_idx, :, 1]
            x, y = x[ra > 0], y[ra > 0]
            xc, yc = x.sum() / ra.sum(), y.sum() / ra.sum()

            # Plot points and center of mass
            axs[i].scatter(x, y, 3, "black", alpha=0.5)
            axs[i].scatter([xc], [yc], 100, "r", "x")
            axs[i].set_aspect("equal", adjustable="datalim")
            axs[i].tick_params(axis="both", which="major", labelsize=6)
            
        # Save plot
        if save_name is None:
            save_name = f"./spot_meridional_{-depth}mm.png"
        else:
            if save_name.endswith('.png'):
                save_name = save_name[:-4]
            save_name = f"{save_name}_meridional_{-depth}mm.png"

        plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close()

    @torch.no_grad()
    def draw_spot_diagram(self, num_fields=5, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """Draw spot diagram of the lens. 
        
        Shot rays from grid points in object space, trace to sensor.

        Args:
            num_fields (int, optional): number of grid points. Defaults to 5.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength of the ray. Defaults to DEFAULT_WAVE.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample and trace rays from grid points
        ray = self.sample_point_source(
            depth=depth,
            num_rays=SPP_PSF,
            num_grid=[num_fields, num_fields],
            wvln=wvln,
        )
        ray = self.trace2sensor(ray)
        ray_o = torch.flip(ray.o.clone(), [0, 1]).cpu().numpy()
        ray_ra = torch.flip(ray.ra.clone(), [0, 1]).cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(num_fields, num_fields, figsize=(num_fields * 2, num_fields * 2))
        for i in range(num_fields):
            for j in range(num_fields):
                ra = ray_ra[i, j, :]
                x, y = ray_o[i, j, :, 0], ray_o[i, j, :, 1]
                x, y = x[ra > 0], y[ra > 0]
                xc, yc = x.sum() / ra.sum(), y.sum() / ra.sum()

                # Plot points and center of mass
                axs[i, j].scatter(x, y, 2, "black", alpha=0.5)
                axs[i, j].scatter([xc], [yc], 100, "r", "x")
                axs[i, j].set_aspect("equal", adjustable="datalim")
                axs[i, j].tick_params(axis='both', which='major', labelsize=6)

        if save_name is None:
            save_name = f"./spot{-depth}mm.png"
        else:
            save_name = f"{save_name}_spot{-depth}mm.png"

        plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close()

    @torch.no_grad()
    def rms_map(self, res=(128, 128), depth=DEPTH):
        """Calculate the RMS spot error map as a weight mask for lens design.

        Args:
            res (tuple, optional): resolution of the RMS map. Defaults to (32, 32).
            depth (float, optional): depth of the point source. Defaults to DEPTH.

        Returns:
            rms_map (torch.Tensor): RMS map normalized to [0, 1].
        """
        ray = self.sample_point_source(depth=depth, num_rays=SPP_PSF, num_grid=64)
        ray, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o2_center = (o2 * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(
            EPSILON
        ).unsqueeze(-1)
        # normalized to center (0, 0)
        o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)

        rms_map = torch.sqrt(
            ((o2_norm**2).sum(-1) * ray.ra).sum(0) / (ray.ra.sum(0) + EPSILON)
        )
        rms_map = (
            F.interpolate(
                rms_map.unsqueeze(0).unsqueeze(0),
                res,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        rms_map /= rms_map.max()

        return rms_map

    # ================================================================
    # PSF
    # ================================================================
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
            psf = self.psf_rgb(points=points[i], ks=ks, center=True, spp=4096)
            psf /= psf.max()

            if log_scale:
                psf = torch.log(psf + EPSILON)
                psf = (psf - psf.min()) / (psf.max() - psf.min())

            psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)

    # ================================================================
    # Distortion
    # ================================================================
    def calc_distortion_2D(
        self, hfov, wvln=DEFAULT_WAVE, plane="meridional", ray_aiming=True
    ):
        """Calculate distortion at a specific field angle.

        Args:
            hfov (float): view angle (degree)
            plane (str): meridional or sagittal
            ray_aiming (bool): whether the chief ray through the center of the stop.

        Returns:
            distortion (float): distortion at the specific field angle
        """
        # Calculate ideal image height
        effective_foclen = self.calc_efl()
        ideal_image_height = effective_foclen * torch.tan(hfov * torch.pi / 180)

        # Calculate chief ray
        chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
            hfov=hfov, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )
        ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)

        ray, _ = self.trace(ray, lens_range=range(len(self.surfaces)))
        t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # Calculate actual image height
        if plane == "sagittal":
            actual_image_height = abs(ray.o[..., 0] + ray.d[..., 0] * t)
        elif plane == "meridional":
            actual_image_height = abs(ray.o[..., 1] + ray.d[..., 1] * t)
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Calculate distortion
        distortion = (actual_image_height - ideal_image_height) / ideal_image_height

        # Handle the case where ideal_image_height is 0 or very close to 0
        mask = abs(ideal_image_height) < EPSILON
        distortion[mask] = torch.tensor(0.0, device=self.device)

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
        distortions = self.calc_distortion_1D(
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

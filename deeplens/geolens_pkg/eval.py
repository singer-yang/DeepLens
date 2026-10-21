# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Classical optical performance evaluation for geometric lens systems.

This module provides a mixin class ``GeoLensEval`` that adds Zemax-equivalent
optical evaluation capabilities to ``GeoLens``.  Spot, distortion and
vignetting metrics are computed purely by geometric ray tracing: rays are
sampled from object space, propagated through all lens surfaces (refraction +
clipping), and analyzed at the sensor plane.  ``mtf`` goes through ``psf()``,
so it follows whichever PSF model that dispatches to, and ``analysis_rendering``
reports PSNR/SSIM from a rendered image.

Coordinate convention (shared with the rest of DeepLens):
    - **z-axis**: optical axis, light travels in +z direction.
    - **y-axis**: meridional (tangential) plane.
    - **x-axis**: sagittal plane.
    - Sensor plane is at ``z = self.d_sensor``.

Key dependencies consumed from the parent ``GeoLens`` instance:
    - ``self.sample_radial_rays()``, ``self.sample_grid_rays()``: ray sampling.
    - ``self.trace(ray)``, ``self.trace2sensor(ray)``: sequential ray tracing.
    - ``self.psf()``, ``self.psf_rgb()``: point-spread-function computation.
    - ``self.render()``: image-plane rendering via ray tracing or PSF convolution.
    - ``self.d_sensor``, ``self.sensor_size``, ``self.pixel_size``, ``self.rfov``,
      ``self.foclen``, ``self.device``: lens geometry attributes.

Functions:
    Spot Diagram:
        spot_points: Core ray-tracing function — samples rays from physical
            object points, traces through the lens, returns sensor positions.
            Shared by draw_spot_radial, draw_spot_map, and rms_map.
        draw_spot_radial: Spot diagrams at evenly-spaced field angles along a
            chosen direction (meridional/sagittal/diagonal), with RGB overlay.
        draw_spot_map: 2-D grid of spot diagrams across the full field of view.

    RMS Spot Error:
        rms_map_rgb: Per-pixel RMS spot radius for R/G/B, referenced to the
            green-channel centroid (chromatic shift included).
        rms_map: Per-pixel RMS spot radius for a single wavelength, referenced
            to its own centroid.

    Distortion:
        calc_distortion_radial: Fractional distortion at evenly-spaced field
            angles along the meridional direction
        draw_distortion_radial: Distortion-vs-field-angle curve (Zemax style).
        calc_distortion_map: 2-D grid of actual-vs-ideal image positions.
        calc_inv_distortion_map: Inverse grid for applying distortion with
            ``grid_sample``.
        draw_distortion_map: Scatter plot of the distortion grid.
        distortion_center: Normalized chief-ray image positions for arbitrary
            object points (used for warp/unwarp).

    MTF (Modulation Transfer Function):
        mtf: Geometric MTF (tangential + sagittal) at a single field position
            via PSF → FFT.
        psf2mtf: Static utility converting a 2-D PSF array to tangential and
            sagittal MTF curves.
        draw_mtf: Grid of tangential MTF curves for multiple depths, FOVs, and
            RGB wavelengths.

    Vignetting:
        vignetting: Fractional ray-throughput map across the field.
        draw_vignetting: Grayscale image of relative illumination.

    Chief Ray:
        calc_chief_ray: Extract the real sampled ray closest to the physical
            aperture-stop centre — from object points (finite conjugates),
            field angles (infinite conjugates), or a traced bundle.

    Comprehensive Analysis:
        analysis_spot: RMS and geometric spot radii and ray throughput over RGB
            at multiple field positions.
        analysis_rendering: Render a test image through the lens and report
            PSNR / SSIM.
        analysis: One-call entry point that chains layout drawing, spot
            analysis, and (optionally) full evaluation + rendering.
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import (
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
)
from ..light import Ray

logger = logging.getLogger(__name__)

# RGB color definitions for wavelength visualization
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    """Mixin that adds classical optical evaluation methods to ``GeoLens``.

    This class is **never instantiated on its own**.  It is mixed into
    ``GeoLens`` via multiple inheritance, so every method can access lens
    geometry (``self.d_sensor``, ``self.rfov``, …) and ray-tracing routines
    (``self.trace()``, ``self.trace2sensor()``, …) directly through ``self``.

    All evaluation functions follow the same pattern:
        1. Sample rays from object space (parallel / grid / radial).
        2. Trace rays through the lens (``self.trace`` or ``self.trace2sensor``).
        3. Analyze ray positions / directions at the sensor plane.
        4. Optionally produce a matplotlib figure saved to disk.

    Results are accuracy-aligned with Zemax OpticStudio for the same lens
    prescriptions and ray-sampling densities.

    Attributes consumed from ``GeoLens`` (via ``self``):
        d_sensor (float): Axial position of the sensor plane (mm).
        sensor_size (tuple[float, float]): Sensor (width, height) in mm.
        pixel_size (float): Pixel pitch in mm.
        sensor_res (tuple[int, int]): Sensor resolution (W, H) in pixels.
        rfov (float): Half field-of-view in **radians**.
        foclen (float): Equivalent focal length in mm.
        fnum (float): F-number.
        aper_idx (int): Index of the aperture stop surface.
        device (torch.device): Compute device (CPU / CUDA).
    """

    # ================================================================
    # Physical-stop chief ray
    # ================================================================
    @torch.no_grad()
    def calc_chief_ray(
        self,
        points_obj=None,
        *,
        ray=None,
        fov=None,
        plane="meridional",
        num_rays=SPP_CALC,
        wvln=None,
        scale_pupil=1.0,
        record=False,
    ):
        """Calculate one real, physical-stop-centred chief ray per field.

        DeepLens uses the classical physical definition: the chief ray is the
        real ray through the centre of the aperture stop. Because the tracer
        samples a finite bundle, this method returns the sampled ray closest to
        the stop centre rather than a synthetic bundle centroid. The selection
        residual scales as ``~1/sqrt(num_rays)`` of the stop radius; raise
        ``num_rays`` when tighter stop centring is needed (e.g. relative
        distortion at small field angles, where the residual is amplified by
        the small ideal image height).

        Exactly one input mode must be supplied:

        1. ``points_obj`` samples and traces rays from object-space points to
           the sensor before extracting the chief ray (finite conjugates).
        2. ``fov`` traces a collimated bundle at the given field angles
           (infinite conjugates) in the ``plane`` direction.
        3. ``ray`` extracts from an already traced bundle that crossed the
           aperture stop, avoiding duplicate tracing.

        Selection happens before final validity is checked. If the closest
        stop-centred ray is clipped after the stop, it remains the selected ray
        and is returned as invalid rather than being replaced by a farther ray.
        The result's ``stop_dist`` is its distance from the stop centre in
        stop radii. If no sample has a finite recorded distance, including an
        untraced bundle, the result is invalid with ``stop_dist=inf``. Multiply
        a recorded distance by the stop radius used for tracing to obtain
        the distance in millimetres.

        Args:
            points_obj (torch.Tensor | list | None): Physical object points in
                millimetres, shape ``[..., 3]``.
            ray (Ray | None): Already traced ray bundle that crossed the
                aperture stop, shape ``[..., num_rays, 3]``.
            fov (float | list | torch.Tensor | None): Field angle(s) in
                degrees for infinite-conjugate chief rays, shape ``[N]`` (a
                scalar becomes ``[1]``).
            plane (str): ``'meridional'`` (y-axis) or ``'sagittal'`` (x-axis);
                used by ``fov`` mode only. Defaults to ``'meridional'``.
            num_rays (int): Samples per field in ``points_obj`` / ``fov``
                mode. Defaults to ``SPP_CALC``.
            wvln (float | None): Wavelength in micrometres for the sampling
                modes. Defaults to the lens primary wavelength.
            scale_pupil (float): Entrance-pupil sampling-radius multiplier in
                the sampling modes. Defaults to 1.0.
            record (bool): Return ``(chief_ray, path)`` with a singleton ray
                dimension in every path entry. Not valid in ``ray`` mode.

        Returns:
            Ray | tuple[Ray, list[torch.Tensor]]: One selected chief ray per
            field, with shape ``[..., 1, 3]``, and optionally its traced path.

        Raises:
            ValueError: If the input modes are invalid, sampling parameters are
                invalid, the supplied bundle is empty, or path
                recording is requested for a supplied bundle.
            TypeError: If ``ray`` is not a :class:`Ray`.

        Note:
            This method is discrete and decorated with ``torch.no_grad()``. It
            is intended for evaluation and PSF centring, not gradient losses.
        """
        has_points = points_obj is not None
        has_ray = ray is not None
        has_fov = fov is not None
        if int(has_points) + int(has_ray) + int(has_fov) != 1:
            raise ValueError("Provide exactly one of points_obj, fov, or ray.")
        if not isinstance(num_rays, int) or num_rays < 1:
            raise ValueError("num_rays must be a positive integer.")
        if not math.isfinite(float(scale_pupil)) or float(scale_pupil) <= 0:
            raise ValueError("scale_pupil must be a finite positive number.")
        if plane not in ("meridional", "sagittal"):
            raise ValueError(f"Invalid plane: {plane}")
        wvln = self.primary_wvln if wvln is None else wvln

        ray_path = None
        if has_points:
            sampled_ray = self.sample_from_points(
                points=points_obj,
                num_rays=num_rays,
                wvln=wvln,
                scale_pupil=scale_pupil,
            )
            if record:
                ray, ray_path = self.trace2sensor(sampled_ray, record=True)
            else:
                ray = self.trace2sensor(sampled_ray)
        elif has_fov:
            fov_t = torch.as_tensor(fov, device=self.device, dtype=self.dtype)
            if fov_t.ndim == 0:
                fov_t = fov_t.unsqueeze(0)
            if fov_t.ndim != 1:
                raise ValueError("fov must be a scalar or a 1-D sequence.")
            if plane == "sagittal":
                fov_x, fov_y = fov_t.tolist(), 0.0
            else:
                fov_x, fov_y = 0.0, fov_t.tolist()
            sampled_ray = self.sample_from_fov(
                fov_x=fov_x,
                fov_y=fov_y,
                depth=float("inf"),
                num_rays=num_rays,
                wvln=wvln,
                scale_pupil=scale_pupil,
            )
            if record:
                ray, ray_path = self.trace2sensor(sampled_ray, record=True)
            else:
                ray = self.trace2sensor(sampled_ray)
        else:
            if not isinstance(ray, Ray):
                raise TypeError("ray must be a deeplens.light.Ray instance.")
            if record:
                raise ValueError(
                    "record=True requires a sampling mode; a supplied traced "
                    "bundle does not contain its earlier path."
                )

        if ray.o.ndim < 2 or ray.o.shape[-2] < 1:
            raise ValueError("ray must contain at least one sample ray.")

        # Rays invalid at the stop carry inf, so the minimum is the closest
        # valid sample and a finite minimum means the field reached the stop.
        stop_dist = ray.stop_dist.masked_fill(ray.stop_dist.isnan(), float("inf"))
        residual_normalized, sample_index = stop_dist.min(dim=-1, keepdim=True)
        stop_reached = torch.isfinite(residual_normalized)

        def gather_scalar(value):
            return torch.gather(value, dim=-1, index=sample_index)

        def gather_vector(value):
            index = sample_index.unsqueeze(-1).expand(
                *sample_index.shape, value.shape[-1]
            )
            return torch.gather(value, dim=-2, index=index)

        chief_ray = ray.clone()
        chief_ray.o = gather_vector(ray.o)
        chief_ray.d = gather_vector(ray.d)
        chief_ray.en = gather_vector(ray.en)
        chief_ray.bend_penalty = gather_vector(ray.bend_penalty)
        chief_ray.opl = gather_vector(ray.opl)
        chief_ray.is_valid = gather_scalar(ray.is_valid) * stop_reached.to(
            dtype=ray.is_valid.dtype
        )
        chief_ray.stop_dist = residual_normalized
        chief_ray.shape = chief_ray.o.shape[:-1]

        if record:
            chief_path = [gather_vector(path_entry) for path_entry in ray_path]
            return chief_ray, chief_path
        return chief_ray

    @torch.no_grad()
    def _chief_or_centroid_xy(self, ray):
        """Per-field transverse reference position from a traced bundle.

        Returns the physical-stop chief-ray position where a valid chief ray
        exists, and the valid-ray centroid elsewhere. The centroid carries a
        coma-induced bias (up to ~0.7% distortion on the sample lenses), so
        the chief ray is preferred whenever it is available.

        Args:
            ray (Ray): Traced bundle with shape ``[..., num_rays, 3]``.

        Returns:
            xy (torch.Tensor): Reference positions, shape ``[..., 2]``.
        """
        centroid_xy = ray.centroid()[..., :2]
        if self.aper_idx is None:
            return centroid_xy

        chief = self.calc_chief_ray(ray=ray)
        chief_xy = chief.o[..., 0, :2]
        valid = chief.is_valid[..., 0] > 0
        if not valid.all():
            logger.warning(
                "%d of %d chief rays are invalid; using the bundle centroid "
                "for those fields.",
                int((~valid).sum().item()),
                valid.numel(),
            )
        return torch.where(valid.unsqueeze(-1), chief_xy, centroid_xy)

    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def spot_points(self, points, num_rays=SPP_PSF, wvln=None):
        """Trace rays from object points to sensor and return the traced Ray.

        Samples rays from each physical object point toward the entrance pupil,
        traces through all lens surfaces (refraction + clipping), and returns
        the resulting Ray object on the sensor plane.

        This is the shared computational core for spot diagrams
        (``draw_spot_radial``, ``draw_spot_map``) and RMS error maps
        (``rms_map``, ``rms_map_rgb``).

        Algorithm:
            1. ``self.sample_from_points(points, num_rays, wvln)`` generates a
               fan of ``num_rays`` rays per object point, aimed at the entrance
               pupil.
            2. ``self.trace2sensor()`` propagates through all surfaces and
               clips vignetted rays.

        Args:
            points (torch.Tensor): Physical 3D object-space coordinates with
                shape ``[..., 3]`` (mm).  Supported layouts:
                - ``[3]`` — single point.
                - ``[N, 3]`` — N points (e.g. radial field positions).
                - ``[H, W, 3]`` — 2-D field grid.
                Generated by
                ``self.point_source_grid(depth=self.obj_depth, normalized=False)``
                for grid sampling, or
                ``self.point_source_radial(depth=self.obj_depth, normalized=False)``
                for radial sampling; ``depth`` is required by both.
            num_rays (int): Number of rays sampled per object point.
                Defaults to ``SPP_PSF``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.

        Returns:
            ray (Ray): Traced ray on the sensor plane, with shape
                ``[..., num_rays, 3]`` for positions and ``[..., num_rays]``
                for validity mask. Use ``ray.o[..., :2]`` for transverse
                positions and ``ray.is_valid`` for the validity mask.
                ``ray.centroid()`` gives the centroid over valid rays.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        ray = self.sample_from_points(points=points, num_rays=num_rays, wvln=wvln)
        return self.trace2sensor(ray)

    @torch.no_grad()
    def draw_spot_radial(
        self,
        save_name="./lens_spot_radial.png",
        num_fov=5,
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        direction="y",
        show=False,
    ):
        """Draw spot diagrams at evenly-spaced field angles along a chosen direction.

        A *spot diagram* visualizes the transverse ray-intercept distribution on
        the sensor plane for a point source at a given field angle and depth.
        It reveals the combined effect of all aberrations (spherical, coma,
        astigmatism, field curvature, chromatic, …).

        Field positions are sampled by **field angle** uniformly from on-axis
        (0) to full-field (``self.rfov``), so the ``FoV 1.0`` subplot reaches
        the full image height — consistent with ``analysis_spot()``.

        Algorithm:
            For each wavelength in ``wvln_list``:
                1. ``self.sample_radial_rays(direction)`` samples rays at
                   ``num_fov`` field angles in ``[0, self.rfov]`` along the
                   chosen direction.
                2. ``self.trace2sensor()`` traces them to the sensor.
                3. Valid ray (x, y) positions are scatter-plotted per subplot.
            All wavelengths are overlaid in a single figure with RGB coloring.

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_spot_radial.png'``.
            num_fov (int): Number of field positions sampled uniformly from
                on-axis (0) to full-field. Defaults to 5.
            depth (float): Object distance in mm (negative = real object).
                When ``None`` or ``float('inf')`` (default ``None``), falls
                back to ``self.obj_depth``.
            num_rays (int): Rays per field position per wavelength.
                Defaults to ``SPP_PSF``.
            wvln_list (list[float]): Wavelengths in µm.  When ``None``
                (default), falls back to ``self.wvln_rgb``.
            direction (str): Sampling direction —
                ``"y"`` (meridional, default), ``"x"`` (sagittal),
                ``"diagonal"`` (45°).
            show (bool): If ``True``, display the figure interactively instead
                of saving to disk. Defaults to ``False``.
        """
        wvln_list = self.wvln_rgb if wvln_list is None else wvln_list
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if depth is None or depth == float("inf"):
            depth = self.obj_depth

        # Field fractions for subplot titles (0 -> on-axis, 1 -> full field)
        fov_fracs = torch.linspace(0, 1, num_fov)

        # Prepare figure
        fig, axs = plt.subplots(1, num_fov, figsize=(num_fov * 3.5, 3))
        axs = np.atleast_1d(axs)

        # Trace and draw each wavelength separately, overlaying results
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays by field angle (0 .. self.rfov) so FoV 1.0 reaches the
            # full image height, matching analysis_spot().
            ray = self.sample_radial_rays(
                num_field=num_fov,
                depth=depth,
                num_rays=num_rays,
                wvln=wvln,
                direction=direction,
                fov_max=self.rfov,
            )
            ray = self.trace2sensor(ray)
            ray_o = ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Plot multiple spot diagrams in one figure
            for i in range(num_fov):
                valid = ray_valid_np[i, :]
                xi, yi = ray_o[i, :, 0], ray_o[i, :, 1]

                # Filter valid rays
                mask = valid > 0
                x_valid, y_valid = xi[mask], yi[mask]

                # Plot points and center of mass for this wavelength
                axs[i].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                axs[i].set_aspect("equal", adjustable="datalim")
                axs[i].tick_params(axis="both", which="major", labelsize=6)
                if wvln_idx == 0:
                    axs[i].set_title(f"FoV {fov_fracs[i].item():.2f}", fontsize=8)

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
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        show=False,
    ):
        """Draw a 2-D grid of spot diagrams across the full field of view.

        Unlike ``draw_spot_radial`` (which samples only a radial slice),
        this method samples a ``num_grid × num_grid`` grid of field positions
        covering both the x (sagittal) and y (meridional) axes, revealing
        off-axis aberrations that are invisible in a 1-D radial scan.

        Grid field positions are sampled by **field angle**, spanning the full
        field on both axes so the corner cells reach the full image height —
        consistent with ``draw_spot_radial`` / ``analysis_spot``.

        Algorithm:
            For each wavelength in ``wvln_list``:
                1. ``self.sample_grid_rays()`` samples a ``grid_h × grid_w``
                   field-angle grid spanning horizontal ``[-hfov/2, hfov/2]``
                   and vertical ``[-vfov/2, vfov/2]`` angles.
                2. ``self.trace2sensor()`` traces them to the sensor.
                3. Valid (x, y) positions are scatter-plotted in the
                   corresponding subplot of the ``num_grid × num_grid`` figure.
            All wavelengths are overlaid with RGB coloring.

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_spot_map.png'``.
            num_grid (int | tuple[int, int]): Number of grid points along each
                axis. Total subplots = ``grid_w * grid_h``. Defaults to 5.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            num_rays (int): Rays per grid cell per wavelength.
                Defaults to ``SPP_PSF``.
            wvln_list (list[float]): Wavelengths in µm.  When ``None``
                (default), falls back to ``self.wvln_rgb``.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        wvln_list = self.wvln_rgb if wvln_list is None else wvln_list
        depth = self.obj_depth if depth is None else depth
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        grid_w, grid_h = num_grid
        fig, axs = plt.subplots(grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3))
        axs = np.atleast_2d(axs)

        # Loop wavelengths and overlay scatters
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample a field-angle grid spanning the full field so the corner
            # cells reach full image height. Shape [grid_h, grid_w, num_rays, 3].
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln
            )
            ray = self.trace2sensor(ray)

            # Convert to numpy, shape [grid_h, grid_w, num_rays, 2]
            ray_o = ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Draw per grid cell
            for i in range(grid_h):
                for j in range(grid_w):
                    valid = ray_valid_np[i, j, :]
                    xi, yi = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                    # Filter valid rays
                    mask = valid > 0
                    x_valid, y_valid = xi[mask], yi[mask]

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
    def rms_map(self, num_grid=32, depth=None, wvln=None, center=None):
        """Compute per-field-position RMS spot radius for a single wavelength.

        Traces ``SPP_PSF`` rays per grid cell and computes the root-mean-square
        distance of valid ray hits from a reference centroid.  When ``center``
        is ``None``, each cell uses its own centroid (monochromatic blur).
        When an external ``center`` is provided (e.g. the green-channel
        centroid), the RMS includes the chromatic shift from that reference.

        Algorithm:
            1. ``self.point_source_grid(depth=depth, grid=num_grid, normalized=False)``
               generates physical object points on a ``[num_grid, num_grid]``
               field grid at the resolved ``depth``.
            2. ``self.spot_points()`` samples ``SPP_PSF`` rays per point and
               traces to sensor.
            3. If ``center`` is ``None``, compute per-cell centroid
               ``c = mean(valid ray_xy)``; otherwise use the provided ``center``.
            4. ``RMS = sqrt( mean( ||ray_xy - c||^2 ) )``.

        Args:
            num_grid (int | tuple[int, int]): Spatial resolution of the field
                sampling grid. Defaults to 32.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            center (torch.Tensor | None): External reference centroid with shape
                ``[grid_h, grid_w, 2]``.  If ``None``, each cell's own
                centroid is used. Defaults to ``None``.

        Returns:
            rms (torch.Tensor): RMS spot error map, shape ``[grid_h, grid_w]``,
                in mm.
            centroid (torch.Tensor): Per-cell centroid used as reference, shape
                ``[grid_h, grid_w, 2]``.  Useful for passing as
                ``center`` to subsequent calls (e.g. in ``rms_map_rgb``).
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Generate physical grid points and trace rays to sensor
        points = self.point_source_grid(depth=depth, grid=num_grid, normalized=False)
        ray = self.spot_points(points, num_rays=SPP_PSF, wvln=wvln)

        # Reuse Ray.centroid() — shape [grid_h, grid_w, 3], slice to [grid_h, grid_w, 2]
        centroid = ray.centroid()[..., :2]

        # Use external center if provided, otherwise own centroid
        ref = center if center is not None else centroid

        # RMS relative to reference, shape [grid_h, grid_w]
        ray_xy = ray.o[..., :2]
        ray_valid = ray.is_valid
        rms = torch.sqrt(
            (((ray_xy - ref.unsqueeze(-2)) ** 2).sum(-1) * ray_valid).sum(-1)
            / (ray_valid.sum(-1) + EPSILON)
        )

        return rms, centroid

    @torch.no_grad()
    def rms_map_rgb(self, num_grid=32, depth=None):
        """Compute per-field-position RMS spot radius for R, G, B wavelengths.

        The RMS spot radius is a standard measure of geometrical image quality.
        For each field position in a ``num_grid × num_grid`` grid, this method
        traces ``SPP_PSF`` rays per wavelength and computes the root-mean-square
        distance of valid ray hits from a **common** reference centroid.

        The reference centroid is the green-channel centroid.  Using a common
        reference means the returned RMS values include *lateral chromatic
        aberration* (the shift between R/G/B centroids), making the map useful
        as a polychromatic image-quality metric.

        Algorithm:
            1. Call ``rms_map(wvln=green)`` to get the green RMS map **and**
               the green centroid.
            2. Call ``rms_map(wvln=red, center=green_centroid)`` and
               ``rms_map(wvln=blue, center=green_centroid)`` to measure R/B
               blur relative to the green reference.
            3. Stack as ``[R, G, B]``.

        Args:
            num_grid (int or tuple[int, int]): Spatial resolution of the field
                sampling grid. Defaults to 32.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.

        Returns:
            rms_rgb (torch.Tensor): RMS spot error map with shape
                ``[3, grid_h, grid_w]`` (channels ordered R, G, B). Units
                are mm (same as sensor coordinates).
        """
        depth = self.obj_depth if depth is None else depth
        # Green first to obtain the shared reference centroid
        rms_g, green_centroid = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[1]
        )

        # Red and blue relative to the green centroid
        rms_r, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[0], center=green_centroid
        )
        rms_b, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[2], center=green_centroid
        )

        return torch.stack([rms_r, rms_g, rms_b], dim=0)

    # ================================================================
    # Distortion
    # ================================================================
    @torch.no_grad()
    def calc_distortion_radial(
        self,
        num_points=GEO_GRID,
        wvln=None,
        plane="meridional",
        num_rays=SPP_CALC,
    ):
        """Compute fractional distortion at evenly-spaced field angles along the meridional direction.

        Distortion is defined as ``(h_actual - h_ideal) / h_ideal``, where
        ``h_ideal = f * tan(theta)`` (rectilinear projection) and ``h_actual``
        is the chief-ray image height on the sensor.  A positive value means
        pincushion distortion; negative means barrel distortion.

        This is the computational counterpart to ``draw_spot_radial``: it
        samples ``num_points`` field angles uniformly from 0 to ``self.rfov``
        and returns both the sampled angles and the corresponding distortion
        values, making it easy to pair with other radial evaluation functions.

        Algorithm:
            1. Derive ``rfov_deg`` from ``self.rfov`` (radians → degrees).
            2. Sample ``num_points`` field angles uniformly in
               ``[0, rfov_deg]``.  The on-axis sample (0°) is replaced by a
               tiny positive angle to avoid 0/0.
            3. Compute ``h_ideal = foclen * tan(angle)`` for each sample.
            4. Compute the physical-stop chief ray at the sensor via
               ``calc_chief_ray(fov=...)``.
            5. Extract ``h_actual`` from the appropriate transverse coordinate
               (x for sagittal, y for meridional).
            6. Return ``(h_actual - h_ideal) / h_ideal``.

        Args:
            num_points (int): Number of evenly-spaced field-angle samples from
                on-axis (0°) to full-field (``self.rfov``).
                Defaults to ``GEO_GRID``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            plane (str): ``'meridional'`` (y-axis) or ``'sagittal'`` (x-axis).
                Defaults to ``'meridional'``.
            num_rays (int): Pupil samples per field angle for chief-ray
                selection. The chief-ray residual (and hence the distortion
                noise floor, amplified at small field angles by the small
                ideal image height) scales as ``~1/sqrt(num_rays)``.
                Defaults to ``SPP_CALC``.

        Returns:
            rfov_samples (np.ndarray): Field angles in degrees, shape
                ``[num_points]``.
            distortions (np.ndarray): Fractional distortion at each angle, shape
                ``[num_points]``.  Dimensionless (multiply by 100 for
                percent).
        """
        wvln = self.primary_wvln if wvln is None else wvln
        rfov_deg = self.rfov * 180 / torch.pi

        # Sample field angles uniformly from 0 to rfov_deg.
        # For the on-axis point (FOV=0), distortion is 0/0.  We compute it at a
        # tiny positive angle to obtain the correct limit, which may be non-zero
        # when the sensor is not at the paraxial focus.
        rfov_samples = torch.linspace(0, rfov_deg, num_points)
        rfov_compute = rfov_samples.clone()
        if rfov_compute[0] == 0:
            # Guard rfov_samples[1] for the single-sample case (num_points == 1).
            tiny = (
                rfov_samples[1].item() * 0.01
                if len(rfov_samples) > 1
                else min(0.01, float(rfov_deg) * 0.01)
            )
            rfov_compute[0] = min(0.01, tiny)

        # Ideal image height: h_ideal = f * tan(theta)
        eff_foclen = float(self.foclen)
        ideal_imgh = eff_foclen * np.tan(rfov_compute.numpy() * np.pi / 180)

        # Actual image height from the physical-stop chief ray at the sensor.
        if plane == "sagittal":
            axis = 0
        elif plane == "meridional":
            axis = 1
        else:
            raise ValueError(f"Invalid plane: {plane}")
        chief_ray = self.calc_chief_ray(
            fov=rfov_compute, plane=plane, wvln=wvln, num_rays=num_rays
        )
        actual_imgh = chief_ray.o[..., 0, axis].abs().cpu().numpy()

        # Fractional distortion, with safe handling of the on-axis singularity
        ideal_imgh = np.asarray(ideal_imgh)
        mask = np.abs(ideal_imgh) < EPSILON
        distortions = np.where(
            mask, 0.0, (actual_imgh - ideal_imgh) / np.where(mask, 1.0, ideal_imgh)
        )

        return rfov_samples.numpy(), distortions

    @torch.no_grad()
    def draw_distortion_radial(
        self,
        save_name=None,
        num_points=GEO_GRID,
        wvln=None,
        plane="meridional",
        show=False,
    ):
        """Draw distortion-vs-field-angle curve in Zemax style.

        Produces a plot with field angle on the y-axis and percent distortion
        on the x-axis, matching the layout convention used in Zemax OpticStudio.
        Useful for quick visual assessment of barrel / pincushion distortion.

        Algorithm:
            1. Call ``calc_distortion_radial`` to obtain field angles and
               fractional distortion values.
            2. Convert distortion to percent and plot.

        Args:
            save_name (str | None): File path for the output PNG.  If ``None``,
                auto-generates ``'./{plane}_distortion_inf.png'``.
            num_points (int): Number of field-angle samples.
                Defaults to ``GEO_GRID``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            plane (str): ``'meridional'`` or ``'sagittal'``.
                Defaults to ``'meridional'``.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        rfov_deg = self.rfov * 180 / torch.pi

        # Calculate distortion at evenly-spaced field angles
        rfov_samples, distortions = self.calc_distortion_radial(
            num_points=num_points, wvln=wvln, plane=plane
        )

        # Convert to percentage and handle NaN
        values = np.nan_to_num(distortions * 100, nan=0.0).tolist()

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
        y_ticks = np.linspace(0, rfov_deg, 3)

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
        ax.set_ylim(0, rfov_deg)

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = f"./{plane}_distortion_inf.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def calc_distortion_map(self, num_grid=16, depth=None, wvln=None):
        """Compute a 2-D distortion grid mapping ideal to actual image positions.

        For each cell in a ``num_grid × num_grid`` field grid, rays are traced
        to the sensor and the physical-stop chief-ray position is extracted
        (fields without a valid chief ray fall back to the bundle centroid).
        The position is then normalized to ``[-1, 1]`` sensor coordinates,
        producing a map that shows how each ideal image point is displaced by
        lens distortion.

        This map can be used with ``torch.nn.functional.grid_sample`` to warp
        or unwarp rendered images.

        Args:
            num_grid (int): Grid resolution along each axis. Defaults to 16.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.

        Returns:
            distortion_grid (torch.Tensor): Distortion grid with shape
                ``[grid_h, grid_w, 2]``. Each entry ``(x, y)`` is in
                normalized sensor coordinates ``[-1, 1]``, representing the
                actual chief-ray image position for the corresponding ideal
                grid position.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # Sample and trace rays, shape (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, wvln=wvln, uniform_fov=False
        )
        ray = self.trace2sensor(ray)

        # Reference position per cell, shape (grid_size, grid_size, 2): the
        # physical-stop chief ray (centroid fallback for invalid fields).
        # Normalize each axis by its own half-extent so non-square sensors
        # map correctly to [-1, 1]: x by sensor_size[0] (width, W),
        # y by sensor_size[1] (height, H).  Sign is flipped on both axes to
        # undo image inversion, matching ``distortion_center``.
        sensor_w, sensor_h = self.sensor_size
        ray_xy = -self._chief_or_centroid_xy(ray)
        x_dist = ray_xy[..., 0] / (sensor_w / 2)
        y_dist = ray_xy[..., 1] / (sensor_h / 2)
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    @torch.no_grad()
    def calc_inv_distortion_map(self, num_grid=16, depth=None, wvln=None):
        """Compute a grid for applying lens distortion with ``grid_sample``.

        For each point on the distorted sensor grid, backward rays are traced
        through the lens to the target object-depth plane. The traced object
        intersections are converted to normalized ideal image coordinates.
        Passing this grid to ``torch.nn.functional.grid_sample`` samples an
        undistorted image and produces a distorted image.

        Args:
            num_grid (int or tuple): Grid resolution. If a tuple is supplied,
                it is interpreted as ``(grid_w, grid_h)``.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.

        Returns:
            inv_distortion_grid (torch.Tensor): Inverse distortion grid with
                shape ``[grid_h, grid_w, 2]`` in ``grid_sample`` coordinates.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        grid_w, grid_h = num_grid
        sensor_w, sensor_h = self.sensor_size
        device = self.device

        # Convert grid_sample output coordinates to physical sensor positions.
        # Distortion maps use negated sensor positions as image coordinates.
        x, y = torch.meshgrid(
            torch.linspace(sensor_w / 2, -sensor_w / 2, grid_w, device=device),
            torch.linspace(sensor_h / 2, -sensor_h / 2, grid_h, device=device),
            indexing="xy",
        )
        z = torch.full_like(x, self.d_sensor.item())

        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(grid_h, grid_w, SPP_CALC)
        )
        ray_o = torch.stack((x, y, z), dim=-1).unsqueeze(2).repeat(1, 1, SPP_CALC, 1)
        ray = Ray(ray_o, ray_o2 - ray_o, wvln, device=device)

        ray = self.trace2obj(ray)
        ray = ray.prop_to(depth)
        # Backward tracing stamps stop weights at the aperture, so the chief
        # backward ray (through the stop centre) gives the object point whose
        # forward chief ray lands on this sensor cell — the exact inverse of
        # the forward distortion map. Centroid fallback for invalid fields.
        point_obj = self._chief_or_centroid_xy(ray)

        scale = self.calc_scale(depth)
        x_ideal = point_obj[..., 0] / (scale * sensor_w / 2)
        y_ideal = point_obj[..., 1] / (scale * sensor_h / 2)
        inv_distortion_grid = torch.stack((x_ideal, y_ideal), dim=-1)
        return inv_distortion_grid

    def distortion_center(self, points):
        """Compute the distorted image position for arbitrary normalized object points.

        Given object points in normalized coordinates, this method converts them
        to physical object-space positions, traces rays from each point through
        the lens, and returns the physical-stop chief-ray position on the sensor
        (bundle-centroid fallback for fields without a valid chief ray) in
        normalized ``[-1, 1]`` coordinates.  This is the inverse mapping needed
        for distortion correction (unwarping).

        Algorithm:
            1. Convert normalized ``(x, y)`` ∈ [-1, 1] to physical object-space
               positions using ``self.calc_scale(depth)`` and ``self.sensor_size``.
            2. ``self.sample_from_points()`` generates rays from each point.
            3. ``self.trace2sensor()`` propagates rays.
            4. Extract the chief-ray position and normalize back to ``[-1, 1]``.

        Args:
            points (torch.Tensor): Normalized point source positions with shape
                ``[N, 3]`` or ``[..., 3]``.  ``x, y`` ∈ [-1, 1] encode the
                field position; ``z`` ∈ (-∞, 0] is the object depth in mm.

        Returns:
            distortion_center (torch.Tensor): Normalized distortion centroid
                positions with shape ``[N, 2]`` or ``[..., 2]``.
                ``x, y`` ∈ [-1, 1].
        """
        sensor_w, sensor_h = self.sensor_size

        # Convert normalized points to object space coordinates
        depth = points[..., 2]
        scale = self.calc_scale(depth)
        points_obj_x = points[..., 0] * scale * sensor_w / 2
        points_obj_y = points[..., 1] * scale * sensor_h / 2
        points_obj = torch.stack([points_obj_x, points_obj_y, depth], dim=-1)

        # Sample rays and trace to sensor
        ray = self.sample_from_points(points=points_obj)
        ray = self.trace2sensor(ray)

        # Chief-ray position (centroid fallback), normalized to [-1, 1]
        ray_center = -self._chief_or_centroid_xy(ray)  # shape [..., 2]
        distortion_center_x = ray_center[..., 0] / (sensor_w / 2)
        distortion_center_y = ray_center[..., 1] / (sensor_h / 2)
        distortion_center = torch.stack(
            (distortion_center_x, distortion_center_y), dim=-1
        )
        return distortion_center

    @torch.no_grad()
    def draw_distortion_map(
        self, save_name=None, num_grid=16, depth=None, wvln=None, show=False
    ):
        """Draw a scatter plot of the distortion grid.

        Visualizes the output of ``calc_distortion_map()`` as a scatter plot on
        ``[-1, 1]`` normalized sensor coordinates.  An undistorted lens would
        show a perfect rectilinear grid; deviations reveal barrel or pincushion
        distortion.

        Args:
            save_name (str | None): File path for the output PNG.  If ``None``,
                auto-generates ``'./distortion_{depth}.png'``.
            num_grid (int): Grid resolution per axis. Defaults to 16.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # Ray tracing to calculate distortion map
        distortion_grid = self.calc_distortion_map(
            num_grid=num_grid, depth=depth, wvln=wvln
        )
        # Scale axes so the plot preserves the physical sensor aspect ratio:
        # longer side → ±1, shorter side → ±(shorter/longer).
        sensor_w, sensor_h = self.sensor_size
        max_half = max(sensor_w, sensor_h) / 2
        aspect_x = (sensor_w / 2) / max_half
        aspect_y = (sensor_h / 2) / max_half
        x1 = distortion_grid[..., 0].cpu().numpy() * aspect_x
        y1 = distortion_grid[..., 1].cpu().numpy() * aspect_y

        # Draw image
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.scatter(x1, y1, s=20, zorder=3)
        ax.axis("scaled")

        # Grid lines based on grid_size, scaled per axis so the overlay
        # matches the data extent (±aspect_x × ±aspect_y).
        ax.set_xticks(np.linspace(-aspect_x, aspect_x, num_grid))
        ax.set_yticks(np.linspace(-aspect_y, aspect_y, num_grid))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

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
    def mtf(self, fov, wvln=None):
        """Compute the geometric MTF at a single field position.

        The *Modulation Transfer Function* describes how well the lens preserves
        contrast as a function of spatial frequency.  MTF = 1 at low frequencies
        (perfect contrast) and falls toward 0 near the diffraction limit or the
        Nyquist frequency of the sensor.

        This implementation uses the *geometric* (ray-based) approach:
            1. Compute the PSF at the given field position via ``self.psf()``.
            2. Convert PSF → MTF via ``psf2mtf()`` (project onto tangential and
               sagittal axes, then take the magnitude of the 1-D FFT).

        Tangential MTF captures resolution in the meridional (radial) direction;
        sagittal MTF captures resolution perpendicular to it.  The difference
        between the two indicates astigmatism.

        Args:
            fov (float): Field angle in radians.  Internally mapped to a
                normalized point ``[0, -fov/rfov, self.obj_depth]``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.

        Returns:
            freq (np.ndarray): Spatial frequency axis in cycles/mm (positive
                frequencies only, excluding DC).
            mtf_tan (np.ndarray): Tangential (meridional) MTF values, normalized
                so that MTF → 1 at low frequency.
            mtf_sag (np.ndarray): Sagittal MTF values, same normalization.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        point = [0, -fov / self.rfov, self.obj_depth]
        psf = self.psf(points=point, recenter=True, wvln=wvln)
        freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size=self.pixel_size)
        return freq, mtf_tan, mtf_sag

    @staticmethod
    def psf2mtf(psf, pixel_size):
        """Convert a 2-D point-spread function to tangential and sagittal MTF curves.

        The MTF is the magnitude of the optical transfer function (OTF), which
        is the Fourier transform of the PSF.  For separable 1-D analysis:
            1. Integrate the PSF along the x-axis → *tangential* line-spread
               function (LSF_tan).
            2. Integrate the PSF along the y-axis → *sagittal* LSF_sag.
            3. Take ``|FFT(LSF)|`` and normalize by the DC component so that
               MTF(0) = 1.

        Only positive frequencies (excluding DC) are returned, following the
        convention used in Zemax MTF plots.

        Args:
            psf (torch.Tensor | np.ndarray): 2-D PSF with shape ``[H, W]``.
                The array's y-axis (rows) corresponds to the **tangential**
                (meridional) direction; x-axis (columns) to the **sagittal**
                direction.
            pixel_size (float): Pixel pitch in mm.  Determines the frequency
                axis scaling: ``Nyquist = 0.5 / pixel_size`` cycles/mm.

        Returns:
            freq (np.ndarray): Spatial frequency in cycles/mm (positive,
                excluding DC).  Length is roughly ``W // 2``.
            mtf_tan (np.ndarray): Tangential MTF, normalized to 1 at DC.
            mtf_sag (np.ndarray): Sagittal MTF, normalized to 1 at DC.

        References:
            - https://en.wikipedia.org/wiki/Optical_transfer_function
            - Edmund Optics: Introduction to Modulation Transfer Function.
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
        depth_list=None,
        psf_ks=128,
        show=False,
    ):
        """Draw a grid of MTF curves for multiple depths and field positions.

        Produces a ``len(depth_list) × len(relative_fov_list)`` subplot grid.
        Each subplot shows the tangential (T, solid) and sagittal (S, dashed)
        MTF for R, G, B wavelengths plus a vertical line at the sensor Nyquist
        frequency (``0.5 / pixel_size`` cycles/mm).

        Algorithm per subplot:
            1. Compute the RGB PSF via ``self.psf_rgb()`` at the specified
               ``(depth, relative_fov)`` with kernel size ``psf_ks``.
            2. For each wavelength channel, call ``psf2mtf()`` to obtain the
               tangential and sagittal MTF curves.
            3. Plot frequency vs MTF with RGB coloring (T solid, S dashed).

        Args:
            save_name (str): File path for the output PNG.
                Defaults to ``'./lens_mtf.png'``.
            relative_fov_list (list[float]): Relative field positions in
                ``[0, 1]``, where 0 = on-axis and 1 = full field.
                Defaults to ``[0.0, 0.7, 1.0]``.
            depth_list (list[float]): Object distances in mm.
                ``float('inf')`` is automatically replaced by
                ``self.obj_depth``.  When ``None`` (default), uses
                ``[self.obj_depth]``.
            psf_ks (int): PSF kernel size in pixels (controls frequency
                resolution of the resulting MTF). Defaults to 128.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        if depth_list is None:
            depth_list = [self.obj_depth]
        pixel_size = self.pixel_size
        nyquist_freq = 0.5 / pixel_size
        num_fovs = len(relative_fov_list)
        if float("inf") in depth_list:
            depth_list = [
                self.obj_depth if x == float("inf") else x for x in depth_list
            ]
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
                psf_rgb = self.psf_rgb(points=point, ks=psf_ks, recenter=True)

                # Calculate MTF curves for rgb wavelengths
                for wvln_idx, wvln in enumerate(self.wvln_rgb):
                    # Calculate tangential + sagittal MTF curves from PSF
                    psf = psf_rgb[wvln_idx]
                    freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size)

                    # Plot MTF curves (tangential solid, sagittal dashed)
                    ax = axs[depth_idx, fov_idx]
                    color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]
                    wvln_label = RGB_LABELS[wvln_idx % len(RGB_LABELS)]
                    wvln_nm = int(wvln * 1000)
                    ax.plot(
                        freq,
                        mtf_tan,
                        color=color,
                        linestyle="-",
                        label=f"{wvln_label}({wvln_nm}nm)-T",
                    )
                    ax.plot(
                        freq,
                        mtf_sag,
                        color=color,
                        linestyle="--",
                        label=f"{wvln_label}({wvln_nm}nm)-S",
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
    # Vignetting
    # ================================================================
    @torch.no_grad()
    def vignetting(self, depth=None, num_grid=32, num_rays=512):
        """Compute the relative-illumination (vignetting) map across the field.

        Vignetting measures how much light is lost at each field position due to
        rays being clipped by lens apertures or barrel edges.  It is computed as
        the fraction of traced rays that remain valid (not vignetted) at each
        grid cell, normalized by the total number of launched rays.

        A value of 1.0 means all rays reach the sensor (no vignetting); 0.0
        means complete light blockage.  Real lenses typically show 1.0 on-axis
        and fall off toward the field edges due to mechanical vignetting and the
        cos⁴ illumination law.

        Algorithm:
            1. ``self.sample_grid_rays()`` with ``uniform_fov=False`` (uniform
               image-space sampling) to ensure correct sensor-plane mapping.
            2. ``self.trace2sensor()`` propagates rays and marks clipped ones as
               invalid.
            3. Per-cell throughput = ``count(valid) / num_rays``.

        Args:
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            num_grid (int): Grid resolution per axis. Defaults to 32.
            num_rays (int): Rays launched per grid cell.  Higher values reduce
                Monte-Carlo noise. Defaults to 512.

        Returns:
            vignetting (torch.Tensor): Vignetting map with shape
                ``[num_grid, num_grid]``, values in ``[0, 1]``.
        """
        depth = self.obj_depth if depth is None else depth
        # Sample rays in uniform image space (not FOV angles) for correct sensor mapping
        # shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=num_rays, uniform_fov=False
        )

        # Trace rays to sensor
        ray = self.trace2sensor(ray)

        # Calculate vignetting map
        vignetting = ray.is_valid.sum(-1) / (ray.is_valid.shape[-1])
        return vignetting

    @torch.no_grad()
    def draw_vignetting(self, filename=None, depth=None, resolution=512, show=False):
        """Draw the vignetting map as a grayscale image with a colorbar.

        Computes the vignetting map via ``self.vignetting()``, bilinearly
        upsamples it to ``resolution × resolution``, and displays it as a
        grayscale image where white = no vignetting and black = fully vignetted.

        Args:
            filename (str | None): File path for the output PNG.  If ``None``,
                auto-generates ``'./vignetting_{depth}.png'``.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            resolution (int): Output image size in pixels (square).
                Defaults to 512.
            show (bool): If ``True``, display interactively. Defaults to ``False``.
        """
        depth = self.obj_depth if depth is None else depth
        # Calculate vignetting map
        vignetting = self.vignetting(depth=depth)

        # Interpolate vignetting map to desired resolution
        vignetting = F.interpolate(
            vignetting.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        fig, ax = plt.subplots()
        ax.set_title("Relative Illumination (Vignetting)")
        im = ax.imshow(vignetting.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

        if show:
            plt.show()
        else:
            if filename is None:
                filename = f"./vignetting_{depth}.png"
            plt.savefig(filename, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ====================================================================================
    # Spot, rendering, and comprehensive analysis
    # ====================================================================================
    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        save_name=None,
        depth=None,
        spp=SPP_RENDER,
        unwarp=False,
        method="ray_tracing",
        show=False,
    ):
        """Render a test image through the lens and report PSNR / SSIM.

        Simulates what the sensor would capture if the given image were placed
        at the specified object distance.  The rendering accounts for all
        geometric aberrations (blur, distortion, vignetting, chromatic effects).
        Optionally applies an inverse distortion warp (``unwarp``) and reports
        quality metrics for both the raw and unwarped renderings.

        Algorithm:
            1. Convert ``img_org`` to a ``[1, 3, H, W]`` float tensor and
               temporarily set the sensor resolution to match.
            2. Call ``self.render()`` with the chosen method (ray tracing or
               PSF-map / PSF-patch convolution).
            3. Compute PSNR and SSIM between the original and rendered images.
            4. If ``unwarp=True``, apply ``self.unwarp()`` to correct geometric
               distortion and report metrics again.
            5. Restore the original sensor resolution.

        Args:
            img_org (np.ndarray | torch.Tensor): Source image with shape
                ``[H, W, 3]``, either uint8 ``[0, 255]`` or float ``[0, 1]``.
            save_name (str | None): Path prefix for saved PNGs.  If not
                ``None``, saves ``'{save_name}.png'`` and (if unwarped)
                ``'{save_name}_unwarped.png'``. Defaults to ``None``.
            depth (float): Object distance in mm. When ``None`` (default),
                falls back to ``self.obj_depth``.
            spp (int): Samples (rays) per pixel for rendering.
                Defaults to ``SPP_RENDER``.
            unwarp (bool): If ``True``, apply distortion correction after
                rendering. Defaults to ``False``.
            method (str): Rendering backend — ``'ray_tracing'``, ``'psf_map'``,
                or ``'psf_patch'``. Defaults to ``'ray_tracing'``.
            show (bool): If ``True``, display the result with matplotlib.
                Defaults to ``False``.

        Returns:
            img_render (torch.Tensor): Rendered (and optionally unwarped) image
                with shape ``[1, 3, H, W]``, float values in ``[0, 1]``.
        """
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from torchvision.utils import save_image

        depth = self.obj_depth if depth is None else depth
        # Change sensor resolution to match the image
        sensor_res_original = self.sensor_res
        if isinstance(img_org, np.ndarray):
            img = (
                torch.from_numpy(img_org).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
        elif torch.is_tensor(img_org):
            img = img_org.permute(2, 0, 1).unsqueeze(0).float()
            if img.max() > 1.0:
                img = img / 255.0
        img = img.to(self.device)
        self.set_sensor_res(sensor_res=img.shape[-2:])

        # Image rendering
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # Compute PSNR and SSIM
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        render_np = (
            img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
        )
        render_psnr = round(
            peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3
        )
        render_ssim = round(
            structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4
        )
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}")

        # Save image
        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # Compute PSNR and SSIM
            render_np = (
                img_render.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 1)
                .cpu()
                .detach()
                .numpy()
            )
            render_psnr = round(
                peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3
            )
            render_ssim = round(
                structural_similarity(
                    img_np, render_np, channel_axis=2, data_range=1.0
                ),
                4,
            )
            print(
                f"Rendered image (unwarped): PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}"
            )

            if save_name is not None:
                save_image(img_render, f"{save_name}_unwarped.png")

        # Change the sensor resolution back
        self.set_sensor_res(sensor_res=sensor_res_original)

        # Show image
        if show:
            plt.imshow(img_render.cpu().squeeze(0).permute(1, 2, 0).numpy())
            plt.title("Rendered image")
            plt.axis("off")
            plt.show()
            plt.close()

        return img_render

    @torch.no_grad()
    def analysis_spot(self, num_field=3, depth=float("inf")):
        """Compute RGB spot radii and ray throughput at multiple field positions.

        Traces rays at ``num_field`` evenly-spaced field positions along the
        meridional direction for three wavelengths (R, G, B), and computes
        polychromatic RMS and geometric spot radii referenced to the
        **combined centroid across all wavelengths** (matching Zemax's
        default "RMS Spot Radius w.r.t. Centroid").

        This provides a quick polychromatic spot-size summary used for design
        comparisons and printed to stdout during ``analysis()``.

        Algorithm (per field point):
            1. Trace R, G, B rays through the lens to the sensor.
            2. Pool all valid ray intercepts (across all three wavelengths)
               and compute one combined centroid ``c``.
            3. RMS = sqrt(mean(||xy - c||²)) over all pooled rays — a single
               polychromatic RMS that includes lateral chromatic aberration.
            4. radius = max(||xy - c||) over all pooled rays.
            5. Convert from mm to μm (× 1000).

        Args:
            num_field (int): Number of field positions sampled from on-axis
                to full-field. Defaults to 3.
            depth (float): Object distance in mm.  Use ``float('inf')`` for
                collimated light. Defaults to ``float('inf')``.

        Returns:
            rms_results (dict[str, dict[str, float]]): Spot analysis results
                keyed by field position string (e.g., ``'fov0.0'``,
                ``'fov0.5'``, ``'fov1.0'``). Each value is a dict with:
                    - ``'rms'``: Polychromatic RMS spot radius in μm.
                    - ``'radius'``: Polychromatic geometric spot radius in μm.
                    - ``'throughput'``: Fraction of sampled rays reaching the sensor;
                      zero marks a lost field even when its spot radius is zero.
        """
        # Trace each wavelength and pool rays across wavelengths per field
        xy_list = []
        valid_list = []
        for wvln in self.wvln_rgb:
            ray = self.sample_radial_rays(
                num_field=num_field,
                depth=depth,
                num_rays=SPP_PSF,
                wvln=wvln,
                fov_max=self.rfov,
            )
            ray = self.trace2sensor(ray)
            xy_list.append(ray.o[..., :2])
            valid_list.append(ray.is_valid)

        # Pool over wavelengths, shape [num_field, 3*num_rays, 2] and [num_field, 3*num_rays]
        xy_all = torch.cat(xy_list, dim=-2)
        valid_all = torch.cat(valid_list, dim=-1)

        # Combined polychromatic centroid per field, shape [num_field, 1, 2]
        valid_mask = valid_all.unsqueeze(-1)
        center = (xy_all * valid_mask).sum(-2) / (
            valid_all.sum(-1, keepdim=True) + EPSILON
        )
        center = center.unsqueeze(-2)

        # Squared distance to combined centroid, shape [num_field, 3*num_rays]
        dist_sq = ((xy_all - center) ** 2).sum(-1)

        # Polychromatic RMS spot radius per field, shape [num_field]
        spot_rms = (
            (dist_sq * valid_all).sum(-1) / (valid_all.sum(-1) + EPSILON)
        ).sqrt()
        # Geometric spot radius (max distance among valid rays)
        dist_masked = torch.where(
            valid_all > 0, dist_sq, torch.full_like(dist_sq, -1.0)
        )
        spot_radius = dist_masked.max(dim=-1).values.clamp(min=0.0).sqrt()

        # Convert mm → μm
        avg_rms_radius_um = spot_rms * 1000.0
        avg_geo_radius_um = spot_radius * 1000.0
        throughput = (valid_all > 0).float().mean(-1)

        # Print results
        print(f"Ray spot analysis results for depth {depth}:")
        print(
            f"RMS radius: FoV (0.0) {avg_rms_radius_um[0]:.3f} um, FoV (0.5) {avg_rms_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_rms_radius_um[-1]:.3f} um"
        )
        print(
            f"Geo radius: FoV (0.0) {avg_geo_radius_um[0]:.3f} um, FoV (0.5) {avg_geo_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_geo_radius_um[-1]:.3f} um"
        )
        print(
            f"Throughput: FoV (0.0) {throughput[0]:.3f}, FoV (0.5) {throughput[num_field // 2]:.3f}, FoV (1.0) {throughput[-1]:.3f}"
        )

        # Save to dict
        rms_results = {}
        fov_ls = torch.linspace(0, 1, num_field)
        for i in range(num_field):
            fov = round(fov_ls[i].item(), 2)
            rms_results[f"fov{fov}"] = {
                "rms": round(avg_rms_radius_um[i].item(), 4),
                "radius": round(avg_geo_radius_um[i].item(), 4),
                "throughput": round(throughput[i].item(), 4),
            }

        return rms_results

    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        depth=float("inf"),
        full_eval=False,
        render=False,
        render_unwarp=False,
        lens_title=None,
        show=False,
    ):
        """Run a comprehensive optical analysis pipeline for the lens.

        This is the main entry point for evaluating a lens design.  It chains
        multiple evaluation steps in order, saving all plots with a common
        ``save_name`` prefix.

        Execution flow:
            1. **Always**: draw the lens layout (``draw_layout``) and compute
               polychromatic spot RMS/radius (``analysis_spot``).
            2. **If** ``full_eval=True``: additionally generate:
               - Spot diagram (``draw_spot_radial``).
               - MTF grid (``draw_mtf``).
               - Distortion curve (``draw_distortion_radial``).
               - Vignetting map (``draw_vignetting``).
            3. **If** ``render=True``: render a test chart image through the
               lens and report PSNR/SSIM (``analysis_rendering``).

        Args:
            save_name (str): Path prefix for all output files.  Each plot
                appends a suffix (e.g., ``'_spot.png'``, ``'_mtf.png'``).
                Defaults to ``'./lens'``.
            depth (float): Object distance in mm.  ``float('inf')`` is replaced
                by ``self.obj_depth`` for rendering and vignetting.
                Defaults to ``float('inf')``.
            full_eval (bool): If ``True``, run all evaluation plots.  If
                ``False``, only layout + spot RMS. Defaults to ``False``.
            render (bool): If ``True``, render a test image through the lens.
                Defaults to ``False``.
            render_unwarp (bool): If ``True`` (and ``render=True``), also
                produce an unwarped rendering. Defaults to ``False``.
            lens_title (str | None): Title string for the layout plot.
                Defaults to ``None``.
            show (bool): If ``True``, display all plots interactively.
                Defaults to ``False``.
        """
        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
            show=show,
        )

        # Calculate RMS error
        self.analysis_spot(depth=depth)

        # Comprehensive optical evaluation
        if full_eval:
            # Draw spot diagram
            self.draw_spot_radial(
                save_name=f"{save_name}_spot.png",
                depth=depth,
                show=show,
            )

            # Draw MTF
            if depth == float("inf"):
                self.draw_mtf(
                    depth_list=[self.obj_depth],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )
            else:
                self.draw_mtf(
                    depth_list=[depth],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )

            # Draw distortion
            self.draw_distortion_radial(
                save_name=f"{save_name}_distortion.png",
                show=show,
            )

            # Draw vignetting
            eval_depth = self.obj_depth if depth == float("inf") else depth
            self.draw_vignetting(
                filename=f"{save_name}_vignetting.png",
                depth=eval_depth,
                show=show,
            )

        # Render an image, compute PSNR and SSIM
        if render:
            depth = self.obj_depth if depth == float("inf") else depth
            img_org = Image.open("./datasets/charts/NBS_1963_1k.png").convert("RGB")
            img_org = np.array(img_org)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                show=show,
            )

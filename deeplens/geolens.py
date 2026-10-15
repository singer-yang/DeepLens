# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Geometric lens model. Differentiable ray tracing is used to simulate light propagation through a geometric lens. Accuracy is aligned with Zemax.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    SPP_CALC,
    SPP_PSF,
    WAVE_RGB,
)
from .geolens_pkg.eval import GeoLensEval
from .geolens_pkg.io import GeoLensIO
from .geolens_pkg.ops import GeoLensOps
from .geolens_pkg.optim import GeoLensOptim
from .geolens_pkg.psf_compute import GeoLensPSF
from .geolens_pkg.render import GeoLensRender
from .geolens_pkg.vis import GeoLensVis
from .geolens_pkg.vis3d import GeoLensVis3D
from .geometric_surface import Aperture
from .lens import Lens
from .light import Ray
from .material import Material

logger = logging.getLogger(__name__)


class GeoLens(
    GeoLensPSF,
    GeoLensRender,
    GeoLensEval,
    GeoLensOptim,
    GeoLensOps,
    GeoLensVis,
    GeoLensIO,
    GeoLensVis3D,
    Lens,
):
    """Differentiable geometric lens using vectorised ray tracing.

    The primary lens model in DeepLens. Supports multi-element refractive
    (and partially reflective) systems loaded from JSON, Zemax `.zmx`, or
    Code V `.seq` files. Accuracy is aligned with Zemax OpticStudio.

    Uses a mixin architecture: eight specialised mixin classes are composed
    at class-definition time to keep each concern isolated: `GeoLensPSF`
    (PSF computation), `GeoLensRender` (image simulation: `render` dispatch
    and reverse ray tracing), `GeoLensEval`
    (spot/MTF/distortion/vignetting evaluation), `GeoLensOptim`
    (losses and gradient-based optimisation), `GeoLensOps` (in-place lens
    operations), `GeoLensVis` (2-D layout/ray visualisation), `GeoLensIO`
    (JSON/Zemax read-write), and `GeoLensVis3D` (3-D mesh visualisation).

    Attributes:
        surfaces (list[Surface]): Ordered list of optical surfaces.
        d_sensor (torch.Tensor): Derived distance from the first surface to the
            sensor plane [mm].
        foclen (float): Effective focal length [mm].
        fnum (float): F-number.
        rfov (float): Real half-diagonal field of view [radians].
        sensor_size (tuple): Physical sensor size (W, H) [mm].
        sensor_res (tuple): Sensor resolution (W, H) [pixels].
        pixel_size (float): Pixel pitch [mm].

    Reference:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," Nature Communications 2024.
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """Initialize a refractive lens.

        There are two ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces

        Args:
            filename (str, optional): Path to lens file (.json, .zmx, or .seq). Defaults to None.
            device (torch.device, optional): Device for tensor computations. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float32.
            primary_wvln (float, optional): Primary design wavelength [µm].
                Used as fallback when a method is called without an explicit
                ``wvln``.  Defaults to ``DEFAULT_WAVE``.
            wvln_rgb (sequence of float, optional): Three wavelengths used
                for RGB computations, ordered ``[R, G, B]`` in µm.  Defaults
                to ``WAVE_RGB``.
            obj_depth (float, optional): Default object depth [mm], used
                when a method is called without an explicit ``depth``.
                Defaults to ``DEPTH``.
        """
        super().__init__(
            device=device,
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )
        # Load lens file
        if filename is not None:
            self.read_lens(filename)
        else:
            self.surfaces = []
            # Placeholder until the caller adds surfaces and calls
            # post_computation(), which sets aper_idx via calc_pupil().
            self.aper_idx = None
            # Set default sensor size and resolution
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            self.to(self.device)

    def read_lens(self, filename):
        """Read a GeoLens from a file.

        Supported file formats:
            - .json: DeepLens native JSON format
            - .zmx: Zemax lens file format
            - .seq: CODE V sequence file format

        Args:
            filename (str): Path to the lens file.

        Note:
            Sensor size and resolution will usually be overwritten by values from the file.
        """
        # Load lens file
        if filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        elif filename[-4:] == ".seq":
            self.read_lens_seq(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # After loading lens, compute foclen, fov and fnum
        self.create_dummy_sensor()
        self.to(self.device)
        self.astype(self.dtype)
        self.post_computation()

    def post_computation(self):
        """Compute derived optical properties after loading or modifying lens.

        Calculates and caches:
            - Effective focal length (EFL)
            - Entrance and exit pupil positions and radii
            - Field of view (FoV) in horizontal, vertical, and diagonal directions
            - F-number
            - Lens design constraints (edge/center thickness bounds, etc.)

        Note:
            This method should be called after any changes to the lens geometry.
        """
        self.calc_foclen()
        self.calc_pupil()
        self.calc_fov()
        self.init_constraints()

    def __call__(self, ray):
        """Trace rays through the lens system (callable shorthand for `trace`).

        Args:
            ray (Ray): Ray object to trace.

        Returns:
            ray_out (Ray): Ray after propagation through the surfaces.
            ray_o_record (list or None): Recorded ray positions, or None.
        """
        return self.trace(ray)

    # ====================================================================================
    # Ray sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_grid_rays(
        self,
        depth=float("inf"),
        num_grid=(11, 11),
        num_rays=SPP_PSF,
        wvln=None,
        uniform_fov=True,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """Sample a grid of rays spanning the field of view from object space.

        If `depth` is infinite, samples collimated rays at evenly-spaced field
        angles; if `depth` is finite, samples diverging point-source rays from a
        grid of object points. Used for PSF maps, RMS error maps, and spot
        diagrams.

        Args:
            depth (float, optional): Object distance in mm. Use `float("inf")`
                for collimated light. Defaults to `float("inf")`.
            num_grid (int or tuple, optional): Number of grid points as
                (num_x, num_y), or a single int for both. Defaults to (11, 11).
            num_rays (int, optional): Number of rays per grid point. Defaults to SPP_PSF.
            wvln (float, optional): Wavelength in µm. When None (default),
                falls back to `self.primary_wvln`.
            uniform_fov (bool, optional): If True, sample uniform FoV angles;
                otherwise sample a uniform object grid. Defaults to True.
            sample_more_off_axis (bool, optional): If True, concentrate grid
                samples toward off-axis fields. Defaults to False.
            scale_pupil (float, optional): Scale factor for pupil radius. Defaults to 1.0.

        Returns:
            rays (Ray): Sampled rays with shape [num_grid[1], num_grid[0], num_rays, 3].
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Normalize num_grid to a tuple if it's an int
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Calculate field angles for grid source. Top-left field has positive fov_x and negative fov_y
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # Calculate FoV_x and FoV_y
        if uniform_fov:
            # Sample uniform FoV angles
            fov_x_list = [x * self.hfov / 2 for x in x_list]
            fov_y_list = [y * self.vfov / 2 for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]
        else:
            # Sample uniform object grid
            fov_x_list = [np.arctan(x * np.tan(self.hfov / 2)) for x in x_list]
            fov_y_list = [np.arctan(y * np.tan(self.vfov / 2)) for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]

        # Sample rays (collimated or point source via unified API)
        rays = self.sample_from_fov(
            fov_x=fov_x_list,
            fov_y=fov_y_list,
            depth=depth,
            num_rays=num_rays,
            wvln=wvln,
            scale_pupil=scale_pupil,
        )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=None,
        direction="y",
        fov_max=None,
    ):
        """Sample radial rays at evenly-spaced field angles along a chosen direction.

        The sampled angles are *radial* field angles: for ``"diagonal"`` the
        per-axis components are ``atan(tan(fov) / sqrt(2))`` so that a field
        listed as ``fov`` really lands at radial angle ``fov``, matching the
        ``"x"`` and ``"y"`` directions.

        Args:
            num_field (int): Number of field angles from on-axis to full-field.
                Defaults to 5.
            depth (float): Object distance in mm. Use ``float('inf')`` for
                collimated light. Defaults to ``float('inf')``.
            num_rays (int): Rays per field position. Defaults to ``SPP_PSF``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            direction (str): Sampling direction —
                ``"y"`` (meridional, default),
                ``"x"`` (sagittal),
                ``"diagonal"`` (45° azimuth).
            fov_max (float): Full-field radial angle [radians]. When ``None``
                (default), falls back to ``self.rfov_eff`` (paraxial pinhole
                FoV). Pass ``self.rfov`` for the real ray-traced FoV, which
                reaches the sensor corner on a distorting lens.

        Returns:
            ray (Ray): Ray object with shape ``[num_field, num_rays, 3]``.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device
        fov_max = self.rfov_eff if fov_max is None else fov_max
        fov_deg = fov_max * 180 / torch.pi
        fov_list = torch.linspace(
            0, fov_deg, num_field, device=device, dtype=self.dtype
        )

        if direction == "y":
            ray = self.sample_from_fov(
                fov_x=0.0, fov_y=fov_list, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "x":
            ray = self.sample_from_fov(
                fov_x=fov_list, fov_y=0.0, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "diagonal":
            # Split the radial angle across both axes so the field lands at
            # radial angle `f`, not at atan(sqrt(2) * tan(f)).
            fov_axis = torch.rad2deg(
                torch.atan(torch.tan(torch.deg2rad(fov_list)) / math.sqrt(2))
            )
            # sample_from_fov creates a meshgrid; for pairwise diagonal, loop
            rays = [
                self.sample_from_fov(
                    fov_x=f.item(),
                    fov_y=f.item(),
                    depth=depth,
                    num_rays=num_rays,
                    wvln=wvln,
                )
                for f in fov_axis
            ]
            ray_o = torch.stack([r.o for r in rays], dim=0)
            ray_d = torch.stack([r.d for r in rays], dim=0)
            ray = Ray(ray_o, ray_d, wvln, device=device)
        else:
            raise ValueError(
                f"Invalid direction: {direction!r}. Use 'x', 'y', or 'diagonal'."
            )
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=None,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample rays from point sources in object space (absolute physical coordinates).

        Rays originate at the given object points and fan out toward the
        entrance pupil. Used for PSF and chief-ray calculation.

        Args:
            points (list or torch.Tensor): Object-space ray origins [mm] with
                shape [3], [N, 3], or [Nx, Ny, 3]. Defaults to [[0.0, 0.0, -10000.0]].
            num_rays (int): Number of rays per point. Defaults to SPP_PSF.
            wvln (float): Wavelength in µm. When None (default), falls back to
                `self.primary_wvln`.
            entrance_pupil (bool): If True (default), aim rays at the entrance
                pupil; otherwise at surface 0.
            scale_pupil (float): Scale factor for pupil radius. Defaults to 1.0.

        Returns:
            rays (Ray): Sampled rays with shape [*points.shape[:-1], num_rays, 3].
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Ray origin is given
        if not torch.is_tensor(points):
            ray_o = torch.as_tensor(points, device=self.device, dtype=self.dtype)
        else:
            ray_o = points.to(device=self.device, dtype=self.dtype)

        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = self.surf_d(0).item(), self.surfaces[0].r
        pupilr *= scale_pupil
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*ray_o.shape[:-1], num_rays)
        )

        # Compute ray directions
        if len(ray_o.shape) == 1:
            # Input point shape is [3]
            ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)  # shape [num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 2:
            # Input point shape is [N, 3]
            ray_o = ray_o.unsqueeze(1).repeat(1, num_rays, 1)  # shape [N, num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 3:
            # Input point shape is [Nx, Ny, 3]
            ray_o = ray_o.unsqueeze(2).repeat(
                1, 1, num_rays, 1
            )  # shape [Nx, Ny, num_rays, 3]
            ray_d = ray_o2 - ray_o

        else:
            raise Exception("The shape of input object positions is not supported.")

        # The physical-stop distance (`ray.stop_dist`) is stamped only when
        # tracing crosses the aperture; entrance-pupil sampling is not the
        # chief-ray definition.
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        return rays

    @torch.no_grad()
    def sample_from_fov(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=float("inf"),
        num_rays=SPP_CALC,
        wvln=None,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample rays from object space at given field angles.

        For infinite depth, generates collimated parallel rays: origins are
        distributed on the entrance pupil and all rays in a field share the
        same direction determined by the FOV angle.

        For finite depth, generates diverging point-source rays: the point
        source position is determined by FOV angle and depth, and rays fan
        out toward the entrance pupil.

        Args:
            fov_x (float or list): Field angle(s) in the xz plane (degrees).
            fov_y (float or list): Field angle(s) in the yz plane (degrees).
            depth (float): Object distance in mm. ``float('inf')`` for
                collimated rays, finite for point-source rays.
            num_rays (int): Number of rays per field point.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            entrance_pupil (bool): If True, sample on entrance pupil;
                otherwise on surface 0. Default: True.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            rays (Ray): Rays with shape ``[..., num_rays, 3]``, where leading dims
                are squeezed when the corresponding fov input is scalar.
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Track which inputs were scalar for output shape
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))
        if x_scalar:
            fov_x = [float(fov_x)]
        if y_scalar:
            fov_y = [float(fov_y)]

        fov_x_rad = torch.as_tensor(fov_x, device=self.device, dtype=self.dtype) * (
            math.pi / 180.0
        )
        fov_y_rad = torch.as_tensor(fov_y, device=self.device, dtype=self.dtype) * (
            math.pi / 180.0
        )
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x_rad, fov_y_rad, indexing="xy")

        if depth == float("inf"):
            # Collimated rays: origins on pupil, uniform direction per field
            if entrance_pupil:
                pupilz, pupilr = self.get_entrance_pupil()
            else:
                pupilz, pupilr = self.surf_d(0).item(), self.surfaces[0].r
            pupilr *= scale_pupil
            ray_o = self.sample_circle(
                r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
            )
            dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
            dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
            dz = torch.ones_like(ray_o[..., 2])
            ray_d = torch.stack((dx, dy, dz), dim=-1)

            if x_scalar:
                ray_o = ray_o.squeeze(1)
                ray_d = ray_d.squeeze(1)
            if y_scalar:
                ray_o = ray_o.squeeze(0)
                ray_d = ray_d.squeeze(0)

            rays = Ray(ray_o, ray_d, wvln, device=self.device)
            rays.prop_to(-1.0)

        else:
            # Point-source rays: origin at object point, fan toward pupil
            x = torch.tan(fov_x_grid) * depth
            y = torch.tan(fov_y_grid) * depth
            z = torch.full_like(x, depth)
            points = torch.stack((x, y, z), dim=-1)

            if x_scalar:
                points = points.squeeze(-2)
            if y_scalar:
                points = points.squeeze(0)

            rays = self.sample_from_points(
                points=points,
                num_rays=num_rays,
                wvln=wvln,
                entrance_pupil=entrance_pupil,
                scale_pupil=scale_pupil,
            )

        return rays

    def sample_circle(self, r, z, shape=[16, 16, 512]):
        """Sample points uniformly inside a circle on a constant-z plane.

        Args:
            r (float): Radius of the circle [mm].
            z (float): Z-coordinate shared by all sampled points [mm].
            shape (list): Shape of the point grid (excluding the trailing
                coordinate dimension). Defaults to [16, 16, 512].

        Returns:
            points (torch.Tensor): Sampled points with shape [*shape, 3].
        """
        device = self.device

        # Generate random angles and radii
        theta = torch.rand(*shape, device=device, dtype=self.dtype) * 2 * torch.pi
        r2 = torch.rand(*shape, device=device, dtype=self.dtype) * r**2
        radius = torch.sqrt(r2)

        # Stack to form 3D points
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)
        points = torch.stack((x, y, z_tensor), dim=-1)

        # Manually sample chief ray
        # points[..., 0, :2] = 0.0

        return points

    # ====================================================================================
    # Ray tracing
    # ====================================================================================
    def trace(self, ray, surf_range=None, record=False):
        """Trace rays through the lens.

        Forward or backward tracing is selected automatically from the sign of
        the ray z-direction.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range, optional): Range of surface indices to trace
                through. When None (default), traces through all surfaces.
            record (bool): If True, record ray positions at each surface. Defaults to False.

        Returns:
            ray_out (Ray): Ray after propagation through the surfaces.
            ray_o_record (list or None): Recorded ray positions at each surface,
                or None when record is False.
        """
        if surf_range is None:
            surf_range = range(0, len(self.surfaces))

        forward = ray.d[..., 2] > 0
        backward = ray.d[..., 2] < 0
        if bool(forward.all().item()):
            ray_out, ray_o_rec = self.forward_tracing(ray, surf_range, record=record)
        elif bool(backward.all().item()):
            ray_out, ray_o_rec = self.backward_tracing(ray, surf_range, record=record)
        else:
            raise ValueError(
                "A ray bundle must have one tracing direction; split mixed "
                "forward/backward or zero-z-direction rays before tracing."
            )

        return ray_out, ray_o_rec

    def trace2obj(self, ray):
        """Trace rays through the lens toward object space.

        Convenience wrapper around `trace` that discards the position record.
        Typically called with sensor-side (backward-propagating) rays.

        Args:
            ray (Ray): Ray object to trace.

        Returns:
            ray (Ray): Ray after propagation through the lens.
        """
        ray, _ = self.trace(ray)
        return ray

    def trace2sensor(self, ray, record=False):
        """Forward trace rays through the lens and propagate them to the sensor plane.

        Args:
            ray (Ray): Ray object to trace.
            record (bool): If True, record ray positions at each surface. Defaults to False.

        Returns:
            ray (Ray): Ray propagated to the sensor plane. When record is True,
                returns a tuple (ray, ray_o_record) where ray_o_record is the list
                of recorded ray positions at each surface (invalid points set to NaN).
        """
        # Trace rays
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.prop_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            # Set to NaN to be skipped in 2d layout visualization
            ray_o[ray.is_valid == 0] = float("nan")
            ray_o_record.append(ray_o)
            return ray, ray_o_record
        else:
            return ray

    def trace2exit_pupil(self, ray):
        """Forward trace rays through the lens to exit pupil plane.

        Args:
            ray (Ray): Ray object to trace.

        Returns:
            ray (Ray): Ray object propagated to the exit pupil plane.
        """
        ray = self.trace2sensor(ray)
        pupil_z, _ = self.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        return ray

    def _validate_surf_range(self, surf_range):
        """Return a bounded contiguous set of surface indices.

        Partial tracing has a well-defined incident medium only for a contiguous
        segment of the sequential prescription. Accepting gaps would propagate
        through skipped vertices without applying their optical interactions.
        """
        surf_indices = [int(i) for i in surf_range]
        if not surf_indices:
            return surf_indices
        if min(surf_indices) < 0 or max(surf_indices) >= len(self.surfaces):
            raise IndexError(
                f"Surface range {surf_indices} is outside [0, {len(self.surfaces) - 1}]."
            )
        expected = list(range(min(surf_indices), max(surf_indices) + 1))
        if sorted(surf_indices) != expected:
            raise ValueError(
                "surf_range must select each surface in one contiguous segment."
            )
        return surf_indices

    def _assign_stop_distance(self, ray):
        """Record each ray's distance from the physical aperture-stop centre.

        The tracing loops call this immediately after the aperture reaction,
        while `ray.o` is on the stop plane. The distance is normalized by the
        stop radius and never enters the autograd graph. Rays invalid at the
        stop receive `inf`, so the minimum identifies the sampled real ray
        closest to the stop centre.
        """
        aper = self.surfaces[self.aper_idx]
        with torch.no_grad():
            dx = ray.o[..., 0] - aper.pos_x
            dy = ray.o[..., 1] - aper.pos_y
            dist = torch.sqrt(dx**2 + dy**2) / max(float(aper.r), EPSILON)
            ray.stop_dist = torch.where(
                torch.isfinite(dist) & (ray.is_valid > 0),
                dist,
                torch.full_like(dist, float("inf")),
            )

    def forward_tracing(self, ray, surf_range, record):
        """Trace forward using sequential per-surface reference frames.

        Rays enter and leave in global coordinates. Interactions happen with
        each vertex at local z=0; after a surface, the ray origin is shifted by
        `-d_next` to express it in the next surface's frame.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            ray_out (Ray): Ray after propagation through all surfaces.
            ray_o_record (list or None): Ray positions at each surface, or None
                if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        surf_indices = self._validate_surf_range(surf_range)
        if not surf_indices:
            return ray, ray_o_record
        first, last = min(surf_indices), max(surf_indices)
        surf_set = set(surf_indices)

        z_frame = self.surf_d(first)
        ray.o[..., 2] -= z_frame

        mat1 = Material("air") if first == 0 else self.surfaces[first - 1].mat2

        # Re-anchor a far object-space bundle once per trace, after entering the
        # first selected surface's local frame: a float32 origin at large |z|
        # loses the low-order sag when the origin and the intersection distance
        # cancel. Anchoring at -10 mm (not -1 mm) keeps rays clear of a concave
        # first surface, whose edge can reach several mm of negative sag.
        if bool((ray.o[..., 2] < -10.0).any()):
            ray.prop_to(-10.0, n=mat1.ior(ray.wvln))

        for i in range(first, last + 1):
            surf = self.surfaces[i]
            if i in surf_set:
                n1 = mat1.ior(ray.wvln)
                n2 = surf.mat2.ior(ray.wvln)
                ray = surf.ray_reaction(ray, n1, n2)
                mat1 = surf.mat2

                if i == self.aper_idx:
                    self._assign_stop_distance(ray)

                if record:
                    ray_out_o = ray.o.clone().detach()
                    ray_out_o[..., 2] += z_frame.detach()
                    ray_out_o[ray.is_valid == 0] = float("nan")
                    ray_o_record.append(ray_out_o)

            dz = surf._get_effective_d_next()
            ray.o[..., 2] -= dz
            z_frame = z_frame + dz

        ray.o[..., 2] += z_frame

        return ray, ray_o_record

    def backward_tracing(self, ray, surf_range, record):
        """Trace backward through the inverse sequential frame steps.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            ray_out (Ray): Ray after backward propagation through all surfaces.
            ray_o_record (list or None): Ray positions at each surface, or None
                if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        surf_indices = self._validate_surf_range(surf_range)
        if not surf_indices:
            return ray, ray_o_record
        first, last = min(surf_indices), max(surf_indices)
        surf_set = set(surf_indices)

        z_frame = self.surf_d(last + 1)
        ray.o[..., 2] -= z_frame

        # The medium on the image side of the last traced surface is always
        # that surface's `mat2`, including when it is the final lens surface.
        # This preserves the pre-refactor material convention for cover glass
        # and other non-air image-space media.
        mat1 = self.surfaces[last].mat2

        # Mirror of the re-anchor in `forward_tracing`: a far float32 origin
        # loses the low-order sag when it cancels against the intersection
        # distance. Local z=0 here is the plane *after* surface `last`, so +10
        # is on the incoming side of every surface in the range. `sample_sensor`
        # enters at local z=0, so the library's own reverse-rendering path never
        # trips this; only a caller-supplied bundle from far +z does.
        if bool((ray.o[..., 2] > 10.0).any()):
            ray.prop_to(10.0, n=mat1.ior(ray.wvln))

        for i in range(last, first - 1, -1):
            surf = self.surfaces[i]
            dz = surf._get_effective_d_next()
            ray.o[..., 2] += dz
            z_frame = z_frame - dz

            if i in surf_set:
                n1 = mat1.ior(ray.wvln)
                mat2 = Material("air") if i == 0 else self.surfaces[i - 1].mat2
                n2 = mat2.ior(ray.wvln)
                ray = surf.ray_reaction(ray, n1, n2)
                mat1 = mat2

                if i == self.aper_idx:
                    self._assign_stop_distance(ray)

                if record:
                    ray_out_o = ray.o.clone().detach()
                    ray_out_o[..., 2] += z_frame.detach()
                    ray_out_o[ray.is_valid == 0] = float("nan")
                    ray_o_record.append(ray_out_o)

        ray.o[..., 2] += z_frame

        return ray, ray_o_record

    # ====================================================================================
    # Geometrical optics calculation
    # ====================================================================================

    @torch.no_grad()
    def calc_foclen(self):
        """Compute effective focal length (EFL) by paraxial ray tracing.

        Traces the paraxial marginal ray of an object at infinity through the
        surfaces with the y-nu recursion, using the reduced slope
        $\\omega = n u$:

        $$\\omega' = \\omega - y \\phi, \\qquad y_{next} = y + \\omega' t / n'$$

        where the surface power $\\phi$ comes from each surface's
        `paraxial_power`. This is the first-order convention used by Zemax and
        CODE V: only vertex geometry contributes, so conic constants, aspheric
        coefficients and freeform departures do not affect the result, and the
        sensor position is irrelevant. Launching $y = 1$, $\\omega = 0$ gives
        $EFL = -n'_k / \\omega'_k$ and $BFL = -y_k n'_k / \\omega'_k$.

        Returns:
            eff_foclen (float): Effective focal length [mm].

        Raises:
            ValueError: If the system is afocal, so the focal length is
                undefined. Failing here avoids writing an infinite `foclen`,
                which would silently poison `calc_fov`/`calc_scale`/`set_fnum`.

        Note:
            Also caches `self.efl` (effective focal length [mm]), `self.foclen`
            (alias of `self.efl`), and `self.bfl` (paraxial back focal length,
            the distance from the last surface to the rear focal point [mm]).

        Reference:
            [1] W. Smith, "Modern Optical Engineering", the y-nu paraxial
                raytrace.
            [2] https://optics.ansys.com/hc/en-us/articles/42661756008083-Understanding-paraxial-ray-tracing
        """
        wvln = self.primary_wvln
        n = Material("air").ior(wvln).double()
        y = torch.ones((), dtype=torch.float64, device=self.device)
        omega = torch.zeros((), dtype=torch.float64, device=self.device)

        for surf in self.surfaces:
            n2 = surf.mat2.ior(wvln).double()
            omega = omega - y * surf.paraxial_power(n, n2, wvln).double()
            y_last = y
            y = y + omega * (surf._get_effective_d_next().double() / n2)
            n = n2

        if omega.abs() < EPSILON:
            raise ValueError(
                "calc_foclen: the system is afocal (zero paraxial power); the "
                "effective focal length is undefined."
            )

        self.efl = float(-n / omega)
        self.foclen = self.efl
        self.bfl = float(-y_last * n / omega)
        return self.efl

    @torch.no_grad()
    def calc_numerical_aperture(self, n=1.0):
        """Compute numerical aperture (NA).

        Args:
            n (float, optional): Refractive index. Defaults to 1.0.

        Returns:
            NA (float): Numerical aperture.

        Reference:
            [1] https://en.wikipedia.org/wiki/Numerical_aperture
        """
        return n * math.sin(math.atan(1 / 2 / self.fnum))
        # return n / (2 * self.fnum)

    @torch.no_grad()
    def calc_focal_plane(self, wvln=None):
        """Compute the focus distance in the object space. Ray starts from sensor center and traces to the object space.

        Args:
            wvln (float, optional): Wavelength in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.

        Returns:
            focal_plane (float): Object-space focus distance [mm] (negative z, in front of the lens).
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device

        # Sample point source rays from sensor center
        o1 = torch.zeros(SPP_CALC, 3, device=device, dtype=self.dtype)
        o1[:, 2] = self.d_sensor

        # Sample the first surface as pupil
        # o2 = self.sample_circle(self.surfaces[0].r, z=0.0, shape=[SPP_CALC])
        # o2 *= 0.5  # Shrink sample region to improve accuracy
        pupilz, pupilr = self.get_exit_pupil()
        o2 = self.sample_circle(pupilr, pupilz, shape=[SPP_CALC])
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=device)

        # Trace rays to object space
        ray = self.trace2obj(ray)

        # Optical axis intersection
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_z = (ray.o[..., 2] - ray.d[..., 2] * t)[ray.is_valid > 0].cpu().numpy()
        focus_z = focus_z[~np.isnan(focus_z) & (focus_z < 0)]

        if len(focus_z) > 0:
            focal_plane = float(np.mean(focus_z))
        else:
            raise ValueError(
                "No valid rays found, focal plane in the image space cannot be computed."
            )

        return focal_plane

    @torch.no_grad()
    def calc_sensor_plane(self, depth=float("inf")):
        """Calculate in-focus sensor plane.

        Args:
            depth (float, optional): Depth of the object plane. Defaults to float("inf").

        Returns:
            d_sensor (torch.Tensor): In-focus sensor z-position [mm] in image space (scalar tensor).
        """
        # Sample and trace rays, shape [SPP_CALC, 3]
        ray = self.sample_from_fov(fov_x=0.0, fov_y=0.0, depth=depth, num_rays=SPP_CALC)
        ray = self.trace2sensor(ray)

        # Calculate in-focus sensor position
        t = (ray.d[:, 0] * ray.o[:, 0] + ray.d[:, 1] * ray.o[:, 1]) / (
            ray.d[:, 0] ** 2 + ray.d[:, 1] ** 2
        )
        focus_z = ray.o[:, 2] - ray.d[:, 2] * t
        focus_z = focus_z[ray.is_valid > 0]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        d_sensor = torch.mean(focus_z)
        return d_sensor

    @torch.no_grad()
    def calc_fov(self):
        """Compute field of view (FoV) of the lens in radians.

        Calculates FoV using two methods:
            1. **Perspective projection** — from focal length and sensor size
               (effective FoV, ignoring distortion).
            2. **Forward ray tracing** — sweeps FOV angles from object side,
               traces to sensor, and interpolates the first angle whose centroid
               reaches the sensor half-diagonal. If a field loses its rays first,
               or no field reaches it, uses the surviving field with the largest
               image height (zero if the on-axis field is unavailable). This
               avoids the failure of the
               old backward-tracing approach on wide-angle lenses where pupil
               aberration at full field leaves zero valid rays.

        Note:
            Caches the following attributes (all FoV values in radians):
            `self.vfov` (vertical FoV), `self.hfov` (horizontal FoV),
            `self.dfov` (diagonal FoV), `self.rfov_eff` (effective paraxial
            half-diagonal FoV, ignoring distortion), `self.rfov` (real
            half-diagonal FoV from ray tracing, accounts for distortion),
            `self.real_dfov` (real diagonal FoV from ray tracing), and
            `self.eqfl` (35 mm equivalent focal length [mm]).

        Reference:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        # 1. Perspective projection (effective FoV)
        self.hfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.vfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov_eff = self.dfov / 2  # effective (paraxial) half-diagonal FoV

        # 2. Forward ray tracing to calculate real FoV (distortion-affected)
        # Sweep FOV angles from object side, trace to sensor, and find which
        # angle produces an image height matching r_sensor.
        num_fov = 64
        fov_hi = min(float(np.rad2deg(self.rfov_eff)) * 1.8, 89.0)
        fov_samples = torch.linspace(
            0.0, fov_hi, num_fov, device=self.device, dtype=self.dtype
        )

        ray = self.sample_from_fov(fov_x=0.0, fov_y=fov_samples.tolist(), num_rays=256)
        ray = self.trace2sensor(ray)

        # Centroid image height per FOV angle, shape [num_fov]
        valid = ray.is_valid > 0  # [num_fov, num_rays]
        masked_y = torch.where(valid, ray.o[..., 1], 0.0)
        n_valid = valid.sum(dim=-1).clamp(min=1)
        imgh = (masked_y.sum(dim=-1) / n_valid).abs()

        # Stop at the first crossing or lost field; a folded outer image must
        # not be mistaken for a larger usable field of view.
        has_valid = valid.sum(dim=-1) > 10
        if has_valid.any():
            beyond = (~has_valid) | (imgh >= self.r_sensor)
            idx = int(beyond.int().argmax()) if beyond.any() else num_fov
            if idx == 0:
                best_deg = fov_samples[0]
            elif idx < num_fov and has_valid[idx]:
                fraction = (self.r_sensor - imgh[idx - 1]) / (
                    imgh[idx] - imgh[idx - 1]
                ).clamp_min(EPSILON)
                best_deg = fov_samples[idx - 1] + fraction * (
                    fov_samples[idx] - fov_samples[idx - 1]
                )
            else:
                # No crossing: stop at the peak image height, since the image
                # folds back beyond it (vignetted outer fields).
                best_deg = fov_samples[int(imgh[:idx].argmax())]
            rfov = math.radians(float(best_deg))
            self.rfov = rfov
            self.real_dfov = 2 * rfov
        else:
            self.rfov = 0.0
            self.real_dfov = 0.0
        if self.rfov == 0.0:
            logger.warning(
                "calc_fov: no usable off-axis field; rfov = 0, so normalized-"
                "field sampling (PSF, spot, MTF) collapses to the axis."
            )

        # 3. Compute 35mm equivalent focal length. 35mm sensor: 36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

    @torch.no_grad()
    def calc_scale(self, depth):
        """Calculate the scale factor (object height / image height).

        Uses the pinhole camera model to compute magnification.

        Args:
            depth (float): Object distance from the lens (negative z direction).

        Returns:
            scale (float): Scale factor relating object height to image height.
        """
        return -depth / self.foclen

    @torch.no_grad()
    def calc_pupil(self):
        """Compute entrance and exit pupil positions and radii.

        The entrance and exit pupils must be recalculated whenever:
            - First-order parameters change (e.g., field of view, object height, image height),
            - Lens geometry or materials change (e.g., surface curvatures, refractive indices, thicknesses),
            - Or generally, any time the lens configuration is modified.

        Note:
            Caches `self.aper_idx` (aperture surface index),
            `self.exit_pupilz`/`self.exit_pupilr` (real exit pupil position and
            radius [mm]), `self.entr_pupilz`/`self.entr_pupilr` (real entrance
            pupil position and radius [mm]),
            `self.exit_pupilz_parax`/`self.exit_pupilr_parax` and
            `self.entr_pupilz_parax`/`self.entr_pupilr_parax` (paraxial pupils),
            and `self.fnum` (F-number from focal length and entrance pupil).
        """
        # Find aperture
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if getattr(self.surfaces[i], "is_aperture", False):
                self.aper_idx = i
                break

        if self.aper_idx is None:
            for i in range(len(self.surfaces)):
                if isinstance(self.surfaces[i], Aperture):
                    self.aper_idx = i
                    break

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])
            print("No aperture found, use the smallest surface as aperture.")

        # Compute entrance and exit pupil
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil_rayaiming()
        self.entr_pupilz, self.entr_pupilr = self.calc_entrance_pupil_rayaiming()
        self.exit_pupilz_parax, self.exit_pupilr_parax = self.calc_pupil_paraxial(
            reverse=False
        )
        self.entr_pupilz_parax, self.entr_pupilr_parax = self.calc_pupil_paraxial(
            reverse=True
        )

        for name, radius in (
            ("entrance", self.entr_pupilr),
            ("exit", self.exit_pupilr),
            ("paraxial entrance", self.entr_pupilr_parax),
            ("paraxial exit", self.exit_pupilr_parax),
        ):
            try:
                radius_value = float(radius)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {name} pupil radius {radius!r}; check lens surfaces "
                    "and aperture data."
                ) from exc
            if not math.isfinite(radius_value) or radius_value <= 0:
                raise ValueError(
                    f"Invalid {name} pupil radius {radius_value!r}; check lens "
                    "surfaces and aperture data."
                )

        # Compute F-number
        self.fnum = self.foclen / (2 * self.entr_pupilr)

    def get_entrance_pupil(self, paraxial=False):
        """Get entrance pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            pupilz (float): Entrance pupil z-position [mm].
            pupilr (float): Entrance pupil radius [mm].
        """
        if paraxial:
            return self.entr_pupilz_parax, self.entr_pupilr_parax
        else:
            return self.entr_pupilz, self.entr_pupilr

    def get_exit_pupil(self, paraxial=False):
        """Get exit pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            pupilz (float): Exit pupil z-position [mm].
            pupilr (float): Exit pupil radius [mm].
        """
        if paraxial:
            return self.exit_pupilz_parax, self.exit_pupilr_parax
        else:
            return self.exit_pupilz, self.exit_pupilr

    @torch.no_grad()
    def calc_pupil_paraxial(self, reverse=False):
        """Image the aperture stop through the surfaces on one side of it.

        The pupils are the first-order images of the stop, so they follow from
        the same y-nu recursion as `calc_foclen` rather than from tracing real
        rays. A ray is launched from the axial point of the stop ($y = 0$,
        $\\omega = n u = 1$) and propagated through the surfaces after the stop
        (exit pupil) or backwards through those before it (entrance pupil);
        where it re-crosses the axis is the pupil plane. Stop and pupil are
        conjugate, so the transverse magnification is given by the
        Smith-Helmholtz relation

        $$m = \\frac{n_0 u_0}{n_k u_k} = \\frac{\\omega_0}{\\omega_k}.$$

        Args:
            reverse (bool, optional): Trace backwards through the surfaces
                before the stop (entrance pupil) instead of forwards through
                those after it (exit pupil). Defaults to False.

        Returns:
            pupilz (float): Pupil z-position [mm].
            pupilr (float): Pupil radius [mm].
        """
        wvln = self.primary_wvln
        air = Material("air").ior(wvln).double()
        surfaces = self.surfaces
        aper_idx = self.aper_idx
        aper_surf = surfaces[aper_idx]
        aper_r = aper_surf.r

        z = self.surf_d(aper_idx).double()
        y = torch.zeros((), dtype=torch.float64, device=self.device)
        omega = torch.ones((), dtype=torch.float64, device=self.device)

        if reverse:
            n = surfaces[aper_idx - 1].mat2.ior(wvln).double() if aper_idx > 0 else air
            for i in range(aper_idx - 1, -1, -1):
                d = surfaces[i]._get_effective_d_next().double()
                y = y - omega * (d / n)
                z = z - d
                n1 = surfaces[i - 1].mat2.ior(wvln).double() if i > 0 else air
                omega = omega + y * surfaces[i].paraxial_power(n1, n, wvln).double()
                n = n1
        else:
            n = aper_surf.mat2.ior(wvln).double()
            for i in range(aper_idx, len(surfaces) - 1):
                d = surfaces[i]._get_effective_d_next().double()
                y = y + omega * (d / n)
                z = z + d
                surf = surfaces[i + 1]
                n2 = surf.mat2.ior(wvln).double()
                omega = omega - y * surf.paraxial_power(n, n2, wvln).double()
                n = n2

        # A stop conjugate to infinity (telecentric side) has no finite pupil.
        if omega.abs() < EPSILON:
            idx = 0 if reverse else -1
            print("Stop images to infinity, use the end surface as pupil.")
            return self.surf_d(idx).item(), surfaces[idx].r

        return float(z - y * n / omega), float(aper_r / omega.abs())

    @torch.no_grad()
    def calc_exit_pupil_rayaiming(self):
        """Calculate exit pupil location and radius from real rays.

        Rays are emitted from the edge of the aperture stop in large quantities
        and traced to the last surface; the exit pupil position and radius come
        from the intersection points of those rays. Slower than
        `calc_pupil_paraxial` and affected by aperture-related aberrations.

        Returns:
            avg_pupilz (float): z coordinate of exit pupil.
            avg_pupilr (float): radius of exit pupil.

        Reference:
            [1] Exit pupil: how many rays can come from sensor to object space.
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        if self.aper_idx is None:
            print("No aperture, use the last surface as exit pupil.")
            return self.surf_d(-1).item(), self.surfaces[-1].r

        # Sample rays from the aperture edge
        aper_z = self.surf_d(self.aper_idx).item()
        aper_r = self.surfaces[self.aper_idx].r
        ray_o = torch.tensor(
            [[aper_r, 0, aper_z]], device=self.device, dtype=self.dtype
        ).repeat(128, 1)
        rfov = float(np.arctan(self.r_sensor / self.foclen))
        phi_rad = torch.linspace(
            -rfov / 2, rfov / 2, 128, device=self.device, dtype=self.dtype
        )

        d = torch.stack(
            (torch.sin(phi_rad), torch.zeros_like(phi_rad), torch.cos(phi_rad)), axis=-1
        )
        ray = Ray(ray_o, d, wvln=self.primary_wvln, device=self.device)

        # Ray tracing from aperture edge to last surface
        surf_range = range(self.aper_idx + 1, len(self.surfaces))
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid != 0][:, 0], ray.o[ray.is_valid != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid != 0][:, 0], ray.d[ray.is_valid != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small pupil
        if len(intersection_points) == 0:
            print("No intersection points found, use the last surface as exit pupil.")
            avg_pupilr = self.surfaces[-1].r
            avg_pupilz = self.surf_d(-1).item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative exit pupil is detected, use the last surface as pupil."
                )
                avg_pupilr = self.surfaces[-1].r
                avg_pupilz = self.surf_d(-1).item()

        return avg_pupilz, avg_pupilr

    @torch.no_grad()
    def calc_entrance_pupil_rayaiming(self):
        """Calculate entrance pupil of the lens from real rays.

        The entrance pupil is the optical image of the physical aperture stop, as seen through the optical elements in front of the stop. We sample backward rays from the aperture stop edge and trace them to the first surface, then find the intersection points of the reverse extension of the rays. The average of the intersection points defines the entrance pupil position and radius. Slower than `calc_pupil_paraxial` and affected by aperture-related aberrations.

        Returns:
            avg_pupilz (float): Entrance pupil z-position [mm].
            avg_pupilr (float): Entrance pupil radius [mm].

        Note:
            [1] Use `calc_pupil_paraxial` unless precise ray aiming is required.
            [2] This function only works for object at a far distance. For microscopes, this function usually returns a negative entrance pupil.

        Reference:
            [1] Entrance pupil: how many rays can come from object space to sensor.
            [2] https://en.wikipedia.org/wiki/Entrance_pupil: "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
            [3] Zemax LLC, *OpticStudio User Manual*, Version 19.4, Document No. 2311, 2019.
        """
        if self.aper_idx is None:
            print("No aperture stop, use the first surface as entrance pupil.")
            return self.surf_d(0).item(), self.surfaces[0].r

        # Sample rays from edge of aperture stop
        aper_z = self.surf_d(self.aper_idx).item()
        aper_r = self.surfaces[self.aper_idx].r

        ray_o = torch.tensor(
            [[aper_r, 0, aper_z]], device=self.device, dtype=self.dtype
        ).repeat(128, 1)
        rfov = float(np.arctan(self.r_sensor / self.foclen))
        phi = torch.linspace(
            -rfov / 2, rfov / 2, 128, device=self.device, dtype=self.dtype
        )

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, wvln=self.primary_wvln, device=self.device)

        # Ray tracing from aperture edge to first surface
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid > 0][:, 0], ray.o[ray.is_valid > 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid > 0][:, 0], ray.d[ray.is_valid > 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small entrance pupil
        if len(intersection_points) == 0:
            print(
                "No intersection points found, use the first surface as entrance pupil."
            )
            avg_pupilr = self.surfaces[0].r
            avg_pupilz = self.surf_d(0).item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface as entrance pupil."
                )
                avg_pupilr = self.surfaces[0].r
                avg_pupilz = self.surf_d(0).item()

        return avg_pupilz, avg_pupilr

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """Compute the intersection points of 2D lines.

        Args:
            origins (torch.Tensor): Origins of the lines. Shape: [N, 2]
            directions (torch.Tensor): Directions of the lines. Shape: [N, 2]

        Returns:
            points (torch.Tensor): Intersection points. Shape: [N*(N-1)/2, 2]
        """
        N = origins.shape[0]

        # Create pairwise combinations of indices
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # Shape: [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # Shape: [N*(N-1)/2, 2]
        Di = directions[idx_i]  # Shape: [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # Shape: [N*(N-1)/2, 2]

        # A pair of non-parallel 2-D lines has a closed-form intersection.
        # Filter parallel/near-parallel pairs before division: batched
        # ``torch.linalg.lstsq`` rejects an entire CUDA batch when even one
        # pair is rank deficient, which made valid real prescriptions fail
        # during pupil estimation.
        b = Oj - Oi  # Shape: [N*(N-1)/2, 2]
        cross_d = Di[:, 0] * Dj[:, 1] - Di[:, 1] * Dj[:, 0]
        direction_scale = torch.linalg.vector_norm(
            Di, dim=-1
        ) * torch.linalg.vector_norm(Dj, dim=-1)
        tolerance = 100 * torch.finfo(directions.dtype).eps * direction_scale
        valid = cross_d.abs() > tolerance
        if not valid.any():
            return origins.new_empty((0, 2))

        Oi = Oi[valid]
        Oj = Oj[valid]
        Di = Di[valid]
        Dj = Dj[valid]
        b = b[valid]
        cross_d = cross_d[valid]
        s = (b[:, 0] * Dj[:, 1] - b[:, 1] * Dj[:, 0]) / cross_d
        t = (b[:, 0] * Di[:, 1] - b[:, 1] * Di[:, 0]) / cross_d

        # Calculate the intersection points using either rays
        P_i = Oi + s.unsqueeze(-1) * Di  # Shape: [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # Shape: [N*(N-1)/2, 2]

        # Take the average to mitigate numerical precision issues
        P = (P_i + P_j) / 2

        return P

    # ====================================================================================
    # Axial geometry derived from sequential surface thicknesses
    # ====================================================================================
    @property
    def d_sensor(self):
        """Global axial position of the sensor plane [mm].

        Surface 0 is the origin. The sensor position is derived by summing each
        surface's `d_next`, so the image plane remains part of the same
        differentiable sequential thickness chain.
        """
        return self.surf_d(len(self.surfaces))

    @d_sensor.setter
    def d_sensor(self, value):
        """Move the sensor by changing the last surface's `d_next` in place."""
        if not self.surfaces:
            raise ValueError("Cannot set d_sensor on a lens without surfaces.")
        target = float(value.detach()) if torch.is_tensor(value) else float(value)
        delta = target - float(self.d_sensor.detach())
        with torch.no_grad():
            self.surfaces[-1].d_next.add_(delta)

    def surf_d(self, idx):
        """Return the derived global vertex position of surface `idx` [mm].

        `surf_d(0)` is zero and `surf_d(i)` is the differentiable prefix sum
        of `d_next` for surfaces before `i`. Passing `len(surfaces)` returns the
        sensor position. Negative indices follow Python surface indexing.
        """
        num_surfs = len(self.surfaces)
        if idx < 0:
            idx += num_surfs
        if idx < 0 or idx > num_surfs:
            raise IndexError(f"Surface index {idx} out of range [0, {num_surfs}].")
        if idx == 0 or num_surfs == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.stack(
            [surface._get_effective_d_next() for surface in self.surfaces[:idx]]
        ).sum()

    # ====================================================================================
    # Lens operation
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, foc_dist=float("inf")):
        """Refocus the lens to a depth distance by changing sensor position.

        Args:
            foc_dist (float, optional): Object focus distance [mm].
                Use ``float('inf')`` for infinity focus. Defaults to ``float('inf')``.

        Note:
            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Calculate in-focus sensor position
        d_sensor_new = self.calc_sensor_plane(depth=foc_dist)

        # Update sensor position
        assert d_sensor_new > 0, "Obtained negative sensor position."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def set_fnum(self, fnum):
        """Set F-number by bracketing the pupil radius, then bisecting.

        Args:
            fnum (float): target F-number.

        Raises:
            ValueError: F-number or target pupil radius is not finite and positive.
            RuntimeError: The pupil target cannot be reached. The original
                aperture radius is restored on failure.
        """
        if not math.isfinite(fnum) or fnum <= 0:
            raise ValueError("F-number must be finite and positive.")
        target_pupil_r = self.foclen / fnum / 2
        if not math.isfinite(target_pupil_r) or target_pupil_r <= 0:
            raise ValueError("Target pupil radius must be finite and positive.")
        aperture = self.surfaces[self.aper_idx]
        original_r = aperture.r
        lo, hi = 0.0, None
        radius = original_r
        try:
            for _ in range(80):
                aperture.update_r(float(radius))
                radius = aperture.r  # update_r enforces the surface's physical limit.
                _, pupilr = self.calc_entrance_pupil_rayaiming()
                if not math.isfinite(pupilr) or pupilr <= 0:
                    break
                if abs(pupilr - target_pupil_r) / target_pupil_r < 1e-3:
                    self.calc_pupil()
                    return
                if pupilr > target_pupil_r:
                    hi = radius
                else:
                    if radius <= lo:
                        break
                    lo = radius
                radius = 2 * radius if hi is None else 0.5 * (lo + hi)
            raise RuntimeError(
                f"Cannot reach F/{fnum}: target pupil radius {target_pupil_r:.6g} mm."
            )
        except Exception:
            # Assign directly: update_r would clamp a stop above max_height().
            aperture.r = original_r
            raise

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov, fnum):
        """Set FoV, image height, and F-number as design targets.

        Only use this method to assign design targets. FoV and focal length
        are overwritten directly; the F-number is reached by resizing the
        aperture stop with `set_fnum`, which ray-traces the entrance pupil.

        Args:
            rfov (float): Half-diagonal FoV. Interpreted as radians; if the
                value is greater than $\\pi$ it is treated as degrees and
                converted to radians.
            fnum (float): Target F-number.

        Raises:
            ValueError: See `set_fnum`.
            RuntimeError: See `set_fnum`. FoV and focal length targets stay
                assigned; only the aperture radius is restored.
        """
        if rfov > math.pi:
            self.rfov_eff = rfov / 180.0 * math.pi
        else:
            self.rfov_eff = rfov

        self.rfov = self.rfov_eff
        self.real_dfov = 2 * self.rfov
        self.foclen = self.r_sensor / math.tan(self.rfov_eff)
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
        # F-number specifies the entrance pupil, not an internal stop's radius.
        self.set_fnum(fnum)

    @torch.no_grad()
    def set_fov(self, rfov):
        """Set half-diagonal field of view as a design target.

        Unlike ``calc_fov()`` which derives FoV from focal length and sensor
        size, this method directly assigns the target FoV for lens optimisation.

        Args:
            rfov (float): Half-diagonal FoV in radians.
        """
        self.rfov_eff = rfov
        self.rfov = rfov
        self.real_dfov = 2 * self.rfov
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

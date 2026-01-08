# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Geometric lens model. Differentiable ray tracing is used to simulate light propagation through a geometric lens. Accuracy is aligned with Zemax.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from deeplens.basics import (
    DEFAULT_WAVE,
    DELTA_PARAXIAL,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
)
from deeplens.geolens_pkg.eval import GeoLensEval
from deeplens.geolens_pkg.io import GeoLensIO
from deeplens.geolens_pkg.optim import GeoLensOptim
from deeplens.geolens_pkg.tolerance import GeoLensTolerance
from deeplens.geolens_pkg.view_3d import GeoLensVis3D
from deeplens.geolens_pkg.vis import GeoLensVis
from deeplens.lens import Lens
from deeplens.optics.geometric_surface import (
    Aperture,
    Aspheric,
    AsphericNorm,
    Cubic,
    Plane,
    Spheric,
    ThinLens,
)
from deeplens.optics.phase_surface import Phase
from deeplens.optics.materials import Material
from deeplens.optics.monte_carlo import forward_integral
from deeplens.optics.ray import Ray
from deeplens.optics.utils import diff_float
from deeplens.optics.wave import AngularSpectrumMethod
from deeplens.utils import (
    batch_psnr,
    batch_ssim,
    img2batch,
)


class GeoLens(
    Lens,
    GeoLensEval,
    GeoLensOptim,
    GeoLensVis,
    GeoLensIO,
    GeoLensTolerance,
    GeoLensVis3D,
):
    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
    ):
        """Initialize a refractive lens.

        There are two ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces and materials

        Args:
            filename (str, optional): Path to lens file (.json, .zmx, or .seq). Defaults to None.
            device (torch.device, optional): Device for tensor computations. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float32.
        """
        super().__init__(device=device, dtype=dtype)

        # Load lens file
        if filename is not None:
            self.read_lens(filename)
        else:
            self.surfaces = []
            self.materials = []
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
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        elif filename[-4:] == ".seq":
            self.read_lens_seq(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # Complete sensor size and resolution if not set from lens file
        if not hasattr(self, "sensor_size"):
            self.sensor_size = (8.0, 8.0)
            print(
                f"Sensor_size not found in lens file. Using default: {self.sensor_size} mm. "
                "Consider specifying sensor_size in the lens file or using set_sensor()."
            )

        if not hasattr(self, "sensor_res"):
            self.sensor_res = (2000, 2000)
            print(
                f"Sensor_res not found in lens file. Using default: {self.sensor_res} pixels. "
                "Consider specifying sensor_res in the lens file or using set_sensor()."
            )
            self.set_sensor_res(self.sensor_res)

        # After loading lens, compute foclen, fov and fnum
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

        Note:
            This method should be called after any changes to the lens geometry.
        """
        self.calc_foclen()
        self.calc_pupil()
        self.calc_fov()

    def __call__(self, ray):
        """Trace rays through the lens system.

        Makes the GeoLens callable, allowing ray tracing with function call syntax.
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
        wvln=DEFAULT_WAVE,
        uniform_fov=True,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """Sample grid rays from object space.
            (1) If depth is infinite, sample parallel rays at different field angles.
            (2) If depth is finite, sample point source rays from the object plane.

        This function is usually used for (1) PSF map, (2) RMS error map, and (3) spot diagram calculation.

        Args:
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_grid (tuple, optional): number of grid points. Defaults to [11, 11].
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            uniform_fov (bool, optional): If True, sample uniform FoV angles.
            sample_more_off_axis (bool, optional): If True, sample more off-axis rays.
            scale_pupil (float, optional): Scale factor for pupil radius.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid[1], num_grid[0], num_rays, 3]
        """
        # Calculate field angles for grid source. Top-left field has positive fov_x and negative fov_y
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # Calculate FoV_x and FoV_y
        if uniform_fov:
            # Sample uniform FoV angles
            fov_x_list = [x * self.vfov / 2 for x in x_list]
            fov_y_list = [y * self.hfov / 2 for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]
        else:
            # Sample uniform object grid
            fov_x_list = [np.atan(x * np.tan(self.vfov / 2)) for x in x_list]
            fov_y_list = [np.atan(y * np.tan(self.hfov / 2)) for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]

        # Sample rays (parallel or point source)
        if depth == float("inf"):
            rays = self.sample_parallel(
                fov_x=fov_x_list,
                fov_y=fov_y_list,
                num_rays=num_rays,
                wvln=wvln,
                scale_pupil=scale_pupil,
            )
        else:
            rays = self.sample_point_source(
                fov_x=fov_x_list,
                fov_y=fov_y_list,
                num_rays=num_rays,
                wvln=wvln,
                depth=depth,
                scale_pupil=scale_pupil,
            )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
    ):
        """Sample radial (meridional, y direction) rays at different field angles.

        This function is usually used for (1) PSF radial map, and (2) RMS error radial map calculation.

        Args:
            num_field (int, optional): number of field angles. Defaults to 5.
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [num_field, num_rays, 3]
        """
        device = self.device
        fov_deg = float(np.rad2deg(self.rfov))
        fov_y_list = torch.linspace(0, fov_deg, num_field, device=device)

        if depth == float("inf"):
            ray = self.sample_parallel(
                fov_x=0.0, fov_y=fov_y_list, num_rays=num_rays, wvln=wvln
            )
        else:
            point_obj_x = torch.zeros(num_field, device=device)
            point_obj_y = depth * torch.tan(fov_y_list * torch.pi / 180.0)
            point_obj = torch.stack(
                [point_obj_x, point_obj_y, torch.full_like(point_obj_x, depth)], dim=-1
            )
            ray = self.sample_from_points(
                points=point_obj, num_rays=num_rays, wvln=wvln
            )
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        scale_pupil=1.0,
    ):
        """
        Sample rays from point sources in object space (absolute physical coordinates).

        Used for PSF and chief ray calculation.

        Args:
            points (list or Tensor): Ray origins in shape [3], [N, 3], or [Nx, Ny, 3].
            num_rays (int): Number of rays per point. Default: SPP_PSF.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Sampled rays with shape [*points.shape[:-1], num_rays, 3].
        """
        # Ray origin is given
        ray_o = torch.tensor(points) if not torch.is_tensor(points) else points
        ray_o = ray_o.to(self.device)

        # Sample points on the pupil
        pupilz, pupilr = self.get_entrance_pupil()
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

        # Calculate rays
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        return rays

    @torch.no_grad()
    def sample_parallel(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        num_rays=SPP_CALC,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
        depth=-1.0,
        scale_pupil=1.0,
    ):
        """
        Sample parallel rays in object space for geometric optics calculations.

        Args:
            fov_x (float or list): Field angle(s) in the xz plane (degrees). Default: [0.0].
            fov_y (float or list): Field angle(s) in the yz plane (degrees). Default: [0.0].
            num_rays (int): Number of rays per field point. Default: SPP_CALC.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            entrance_pupil (bool): If True, sample origins on entrance pupil; otherwise, on surface 0. Default: True.
            depth (float): Propagation depth in z. Default: -1.0.
            scale_pupil (float): Scale factor for pupil radius. Default: 1.0.

        Returns:
            Ray:
                Rays with shape [..., num_rays, 3], where leading dims are:
                - both fov_x and fov_y scalars: [num_rays, 3]
                - fov_x scalar: [len(fov_y), num_rays, 3]
                - fov_y scalar: [len(fov_x), num_rays, 3]
                - both lists: [len(fov_y), len(fov_x), num_rays, 3]
                Ordered as (u, v).
        """
        # Remember whether inputs were scalar
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))

        # Normalize to lists for internal processing
        if x_scalar:
            fov_x = [float(fov_x)]
        if y_scalar:
            fov_y = [float(fov_y)]

        fov_x = torch.tensor([fx * torch.pi / 180 for fx in fov_x]).to(self.device)
        fov_y = torch.tensor([fy * torch.pi / 180 for fy in fov_y]).to(self.device)

        # Sample ray origins on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
            pupilr *= scale_pupil
        else:
            pupilz, pupilr = 0.0, self.surfaces[0].r
            pupilr *= scale_pupil

        ray_o = self.sample_circle(
            r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
        )

        # Sample ray directions
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x, fov_y, indexing="xy")
        dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
        dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
        dz = torch.ones_like(ray_o[..., 2])
        ray_d = torch.stack((dx, dy, dz), dim=-1)

        # Squeeze singleton FOV dims only if the original input was scalar
        if x_scalar:
            ray_o = ray_o.squeeze(1)
            ray_d = ray_d.squeeze(1)
        if y_scalar:
            ray_o = ray_o.squeeze(0)
            ray_d = ray_d.squeeze(0)

        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=DEPTH,
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample point source rays from object space with given field angles.

        Used for (1) spot/rms/magnification calculation, (2) distortion/sensor sampling.

        This function is equivalent to self.point_source_grid() + self.sample_from_points().

        Args:
            fov_x (float or list): field angle in x0z plane.
            fov_y (float or list): field angle in y0z plane.
            depth (float, optional): sample plane z position. Defaults to -10.0.
            num_rays (int, optional): number of rays sampled from each grid point. Defaults to 16.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [len(fov_y), len(fov_x), num_rays, 3], arranged in uv order.
        """
        # Sample second points on the pupil, shape [len(fov_y), len(fov_x), num_rays, 3]
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
            pupilr *= scale_pupil
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        # Sample grid points with given field angles, shape [len(fov_y), len(fov_x), 3]
        fov_x = torch.tensor([fx * torch.pi / 180 for fx in fov_x]).to(self.device)
        fov_y = torch.tensor([fy * torch.pi / 180 for fy in fov_y]).to(self.device)
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x, fov_y, indexing="xy")
        x, y = torch.tan(fov_x_grid) * depth, torch.tan(fov_y_grid) * depth

        # Form ray origins, shape [len(fov_y), len(fov_x), num_rays, 3]
        z = torch.full_like(x, depth)
        ray_o = torch.stack((x, y, z), -1)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, num_rays, 1)

        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(len(fov_y), len(fov_x), num_rays)
        )

        # Compute ray directions
        ray_d = ray_o2 - ray_o

        ray = Ray(ray_o, ray_d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=DEFAULT_WAVE, sub_pixel=False):
        """Sample rays from sensor pixels (backward rays). Used for ray tracing rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [H, W, spp, 3]
        """
        w, h = self.sensor_size
        W, H = self.sensor_res
        device = self.device

        # Sample points on sensor plane
        # Use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(
                -w / 2,
                w / 2,
                W + 1,
                device=device,
            )[1:],
            torch.linspace(
                h / 2,
                -h / 2,
                H + 1,
                device=device,
            )[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor.item())

        # Sample second points on the pupil
        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(*self.sensor_res, spp))

        # Form rays
        ray_o = torch.stack((x1, y1, z1), 2)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, spp, 1)  # [H, W, spp, 3]

        # Sub-pixel sampling for more realistic rendering
        if sub_pixel:
            delta_ox = (
                torch.rand((ray_o[:, :, :, 0].clone().shape), device=device)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand((ray_o[:, :, :, 1].clone().shape), device=device)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # Form rays
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=device)
        return ray

    def sample_circle(self, r, z, shape=[16, 16, 512]):
        """Sample points inside a circle.

        Args:
            r (float): Radius of the circle.
            z (float): Z-coordinate for all sampled points.
            shape (list): Shape of the output tensor.

        Returns:
            torch.Tensor: Tensor of shape [*shape, 3] containing sampled points.
        """
        device = self.device

        # Generate random angles and radii
        theta = torch.rand(*shape, device=device) * 2 * torch.pi
        r2 = torch.rand(*shape, device=device) * r**2
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

        Forward or backward tracing is automatically determined by the ray direction.

        Args:
            ray (Ray object): Ray object.
            surf_range (list): Surface index range.
            record (bool): record ray path or not.

        Returns:
            ray_final (Ray object): ray after optical system.
            ray_o_rec (list): list of intersection points.
        """
        if surf_range is None:
            surf_range = range(0, len(self.surfaces))

        if (ray.d[..., 2].unsqueeze(-1) > 0).any():
            ray_out, ray_o_rec = self.forward_tracing(ray, surf_range, record=record)
        else:
            ray_out, ray_o_rec = self.backward_tracing(ray, surf_range, record=record)

        return ray_out, ray_o_rec

    def trace2obj(self, ray):
        """Traces rays backwards through all lens surfaces from sensor side
        to object side.

        Args:
            ray (Ray): Ray object to trace backwards.

        Returns:
            Ray: Ray object after backward propagation through the lens.
        """
        ray, _ = self.trace(ray)
        return ray

    def trace2sensor(self, ray, record=False):
        """Forward trace rays through the lens to sensor plane.

        Args:
            ray (Ray object): Ray object.
            record (bool): record ray path or not.

        Returns:
            ray_out (Ray object): ray after optical system.
            ray_o_record (list): list of intersection points.
        """
        # Manually propagate ray to a shallow depth to avoid numerical instability
        if (ray.o[..., 2].min() < -100.0).any():
            ray = ray.prop_to(-10.0)

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
            Ray: Ray object propagated to the exit pupil plane.
        """
        ray = self.trace2sensor(ray)
        pupil_z, _ = self.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        return ray

    def forward_tracing(self, ray, surf_range, record):
        """Forward traces rays through each surface in the specified range from object side to image side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in surf_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    def backward_tracing(self, ray, surf_range, record):
        """Backward traces rays through each surface in reverse order from image side to object side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after backward propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in np.flip(surf_range):
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i - 1].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i - 1].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    # ====================================================================================
    # Image simulation
    # ====================================================================================
    def render(self, img_obj, depth=DEPTH, method="ray_tracing", **kwargs):
        """Differentiable image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] PSF patch convolution.
            [3] Ray tracing rendering.

        Args:
            img_obj (Tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. One of 'psf_map', 'psf_patch',
                or 'ray_tracing'. Defaults to 'ray_tracing'.
            **kwargs: Additional arguments for different methods:
                - psf_grid (tuple): Grid size for PSF map method. Defaults to (10, 10).
                - psf_ks (int): Kernel size for PSF methods. Defaults to PSF_KS.
                - psf_center (tuple): Center position for PSF patch method.
                - spp (int): Samples per pixel for ray tracing. Defaults to SPP_RENDER.

        Returns:
            Tensor: Rendered image tensor. Shape of [N, C, H, W].
        """
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # Image simulation
        if method == "psf_map":
            # PSF rendering - uses PSF map to render image
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_map(
                img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
            )

        elif method == "psf_patch":
            # PSF patch rendering - uses a single PSF to render a patch of the image
            psf_center = kwargs.get("psf_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
            )

        elif method == "ray_tracing":
            # Ray tracing rendering
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            spp = kwargs.get("spp", SPP_RENDER)
            img_render = self.render_raytracing(img_obj, depth=depth, spp=spp)

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_raytracing(self, img, depth=DEPTH, spp=SPP_RENDER, vignetting=False):
        """Render RGB image using ray tracing rendering.

        Args:
            img (tensor): RGB image tensor. Shape of [N, 3, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to 64.
            vignetting (bool, optional): whether to consider vignetting effect. Defaults to False.

        Returns:
            img_render (tensor): Rendered RGB image tensor. Shape of [N, 3, H, W].
        """
        img_render = torch.zeros_like(img)
        for i in range(3):
            img_render[:, i, :, :] = self.render_raytracing_mono(
                img=img[:, i, :, :],
                wvln=WAVE_RGB[i],
                depth=depth,
                spp=spp,
                vignetting=vignetting,
            )
        return img_render

    def render_raytracing_mono(self, img, wvln, depth=DEPTH, spp=64, vignetting=False):
        """Render monochrome image using ray tracing rendering.

        Args:
            img (tensor): Monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
            wvln (float): Wavelength of the light.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to 64.

        Returns:
            img_mono (tensor): Rendered monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
        """
        img = torch.flip(img, [-2, -1])
        scale = self.calc_scale(depth=depth)
        ray = self.sample_sensor(spp=spp, wvln=wvln)
        ray = self.trace2obj(ray)
        img_mono = self.render_compute_image(
            img, depth=depth, scale=scale, ray=ray, vignetting=vignetting
        )
        return img_mono

    def render_compute_image(self, img, depth, scale, ray, vignetting=False):
        """Computes the intersection points between rays and the object image plane, then generates the rendered image following rendering equation.

        Back-propagation gradient flow: image -> w_i -> u -> p -> ray -> surface

        Args:
            img (tensor): [N, C, H, W] or [N, H, W] shape image tensor.
            depth (float): depth of the object.
            scale (float): scale factor.
            ray (Ray object): Ray object. Shape [H, W, spp, 3].
            vignetting (bool): whether to consider vignetting effect.

        Returns:
            image (tensor): [N, C, H, W] or [N, H, W] shape rendered image tensor.
        """
        assert torch.is_tensor(img), "Input image should be Tensor."

        # Padding
        H, W = img.shape[-2:]
        if len(img.shape) == 3:
            img = F.pad(img.unsqueeze(1), (1, 1, 1, 1), "replicate").squeeze(1)
        elif len(img.shape) == 4:
            img = F.pad(img, (1, 1, 1, 1), "replicate")
        else:
            raise ValueError("Input image should be [N, C, H, W] or [N, H, W] tensor.")

        # Scale object image physical size to get 1:1 pixel-pixel alignment with sensor image
        ray = ray.prop_to(depth)
        p = ray.o[..., :2]
        pixel_size = scale * self.pixel_size
        ray.is_valid = (
            ray.is_valid
            * (torch.abs(p[..., 0] / pixel_size) < (W / 2 + 1))
            * (torch.abs(p[..., 1] / pixel_size) < (H / 2 + 1))
        )

        # Convert to uv coordinates in object image coordinate
        # (we do padding so corrdinates should add 1)
        u = torch.clamp(W / 2 + p[..., 0] / pixel_size, min=-0.99, max=W - 0.01)
        v = torch.clamp(H / 2 + p[..., 1] / pixel_size, min=0.01, max=H + 0.99)

        # (idx_i, idx_j) denotes left-top pixel (reference pixel). Index does not store gradients
        # (idx + 1 because we did padding)
        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1

        # Gradients are stored in interpolation weight parameters
        w_i = v - v.floor().long()
        w_j = u.ceil().long() - u

        # Bilinear interpolation
        # (img shape [B, N, H', W'], idx_i shape [H, W, spp], w_i shape [H, W, spp], irr_img shape [N, C, H, W, spp])
        irr_img = img[..., idx_i, idx_j] * w_i * w_j
        irr_img += img[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
        irr_img += img[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
        irr_img += img[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)

        # Computation image
        if not vignetting:
            image = torch.sum(irr_img * ray.is_valid, -1) / (
                torch.sum(ray.is_valid, -1) + EPSILON
            )
        else:
            image = torch.sum(irr_img * ray.is_valid, -1) / torch.numel(ray.is_valid)

        return image

    def unwarp(self, img, depth=DEPTH, num_grid=128, crop=True, flip=True):
        """Unwarp rendered images using distortion map.

        Args:
            img (tensor): Rendered image tensor. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            grid_size (int, optional): Grid size. Defaults to 256.
            crop (bool, optional): Whether to crop the image. Defaults to True.

        Returns:
            img_unwarpped (tensor): Unwarped image tensor. Shape of [N, C, H, W].
        """
        # Calculate distortion map, shape (num_grid, num_grid, 2)
        distortion_map = self.distortion_map(depth=depth, num_grid=num_grid)

        # Interpolate distortion map to image resolution
        distortion_map = distortion_map.permute(2, 0, 1).unsqueeze(1)
        # distortion_map = torch.flip(distortion_map, [-2]) if flip else distortion_map
        distortion_map = F.interpolate(
            distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )  # shape (B, 2, Himg, Wimg)
        distortion_map = distortion_map.permute(1, 2, 3, 0).repeat(
            img.shape[0], 1, 1, 1
        )  # shape (B, Himg, Wimg, 2)

        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(
            img, distortion_map, align_corners=True
        )  # shape (B, C, Himg, Wimg)
        return img_unwarpped

    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        save_name=None,
        depth=DEPTH,
        spp=SPP_RENDER,
        unwarp=False,
        noise=0.0,
        method="ray_tracing",
        show=False,
    ):
        """Render a single image for visualization and analysis.

        Args:
            img_org (Tensor): Original image with shape [H, W, 3].
            save_name (str, optional): Path prefix for saving rendered images. Defaults to None.
            depth (float, optional): Depth of object image. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to SPP_RENDER.
            unwarp (bool, optional): If True, unwarp the image to correct distortion. Defaults to False.
            noise (float, optional): Gaussian noise standard deviation. Defaults to 0.0.
            method (str, optional): Rendering method ('ray_tracing', etc.). Defaults to 'ray_tracing'.
            show (bool, optional): If True, display the rendered image. Defaults to False.

        Returns:
            Tensor: Rendered image tensor with shape [1, 3, H, W].
        """
        # Change sensor resolution to match the image
        sensor_res_original = self.sensor_res
        img = img2batch(img_org).to(self.device)
        self.set_sensor_res(sensor_res=img.shape[-2:])

        # Image rendering
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # Add noise (a very simple Gaussian noise model)
        if noise > 0:
            img_render = img_render + torch.randn_like(img_render) * noise
            img_render = torch.clamp(img_render, 0, 1)

        # Compute PSNR and SSIM
        render_psnr = round(batch_psnr(img, img_render).item(), 3)
        render_ssim = round(batch_ssim(img, img_render).item(), 3)
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.3f}")

        # Save image
        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # Compute PSNR and SSIM
            render_psnr = round(batch_psnr(img, img_render).item(), 3)
            render_ssim = round(batch_ssim(img, img_render).item(), 3)
            print(
                f"Rendered image (unwarped): PSNR={render_psnr:.3f}, SSIM={render_ssim:.3f}"
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

    # ====================================================================================
    # PSF
    # We support three types of PSF:
    #   1. Geometric PSF (`psf`): incoherent intensity ray tracing
    #   2. Exit-pupil PSF (`psf_pupil_prop` / `psf_coherent`): coherent ray tracing to exit pupil, then free-space propagation with ASM
    #   3. Huygens PSF (`psf_huygens`): coherent ray tracing to exit pupil, then Huygens-Fresnel integration
    # ====================================================================================
    def psf(
        self,
        points,
        ks=PSF_KS,
        wvln=DEFAULT_WAVE,
        spp=None,
        recenter=True,
        model="geometric",
    ):
        """Calculate Point Spread Function (PSF) for given point sources.

        Supports multiple PSF calculation models:
            - geometric: Incoherent intensity ray tracing (fast, differentiable)
            - coherent: Coherent ray tracing with free-space propagation (accurate, differentiable)
            - huygens: Huygens-Fresnel integration (accurate, not differentiable)

        Args:
            points (Tensor): Point source positions. Shape [N, 3] with x, y in [-1, 1]
                and z in [-Inf, 0]. Normalized coordinates.
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            wvln (float, optional): Wavelength in [um]. Defaults to DEFAULT_WAVE.
            spp (int, optional): Samples per pixel. If None, uses model-specific default.
            recenter (bool, optional): If True, center PSF using chief ray. Defaults to True.
            model (str, optional): PSF model type. One of 'geometric', 'coherent', 'huygens'.
                Defaults to 'geometric'.

        Returns:
            Tensor: PSF normalized to sum to 1. Shape [ks, ks] or [N, ks, ks].
        """
        if model == "geometric":
            spp = SPP_PSF if spp is None else spp
            return self.psf_geometric(points, ks, wvln, spp, recenter)
        elif model == "coherent":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_coherent(points, ks, wvln, spp, recenter)
        elif model == "huygens":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_huygens(points, ks, wvln, spp, recenter)
        else:
            raise ValueError(f"Unknown PSF model: {model}")

    def psf_geometric(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_PSF, recenter=True
    ):
        """Single wavelength geometric PSF calculation.

        Args:
            points (Tensor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            ks (int, optional): Output kernel size.
            wvln (float, optional): Wavelength.
            spp (int, optional): Sample per pixel.
            recenter (bool, optional): Recenter PSF using chief ray.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].

        References:
            [1] https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function
        """
        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # Sample rays. Ray position in the object space by perspective projection
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays to sensor plane (incoherent)
        ray.coherent = False
        ray = self.trace2sensor(ray)

        # Calculate PSF center, shape [N, 2]
        if recenter:
            pointc = self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = self.psf_center(point_obj, method="pinhole")

        # Monte Carlo integration
        psf = forward_integral(ray.flip_xy(), ps=pixel_size, ks=ks, pointc=pointc)

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_coherent(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Alias for psf_pupil_prop. Calculates PSF by coherent ray tracing to exit pupil followed by Angular Spectrum Method (ASM) propagation."""
        return self.psf_pupil_prop(points, ks=ks, wvln=wvln, spp=spp, recenter=recenter)

    def psf_pupil_prop(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Single point monochromatic PSF using exit-pupil diffraction model. This function is differentiable.

        Steps:
            1, Calculate complex wavefield at exit-pupil plane by coherent ray tracing.
            2, Free-space propagation to sensor plane and calculate intensity PSF.

        Args:
            points (torch.Tensor, optional): [x, y, z] coordinates of the point source. Defaults to torch.Tensor([0,0,-10000]).
            ks (int, optional): size of the PSF patch. Defaults to PSF_KS.
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            spp (int, optional): number of rays to sample. Defaults to SPP_COHERENT.
            recenter (bool, optional): Recenter PSF using chief ray. Defaults to True.

        Returns:
            psf_out (torch.Tensor): PSF patch. Normalized to sum to 1. Shape [ks, ks]

        Reference:
            [1] "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model", SIGGRAPH Asia 2024.

        Note:
            [1] This function is similar to ZEMAX FFT_PSF but implement free-space propagation with Angular Spectrum Method (ASM) rather than FFT transform. Free-space propagation using ASM is more accurate than doing FFT, because FFT (as used in ZEMAX) assumes far-field condition (e.g., chief ray perpendicular to image plane).
        """
        # Pupil field by coherent ray tracing
        wavefront, psfc = self.pupil_field(
            points=points, wvln=wvln, spp=spp, recenter=recenter
        )

        # Propagate to sensor plane and get intensity
        pupilz, pupilr = self.get_exit_pupil()
        h, w = wavefront.shape
        # Manually pad wave field
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        # Free-space propagation using Angular Spectrum Method (ASM)
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=self.d_sensor - pupilz,
            wvln=wvln,
            ps=self.pixel_size,
            padding=False,
        )
        # Get intensity
        psf_inten = sensor_field.abs() ** 2

        # Calculate PSF center
        h, w = psf_inten.shape[-2:]
        # consider both interplation and padding
        psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
        psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

        # Crop valid PSF region and normalize
        if ks is not None:
            psf_inten_pad = (
                F.pad(
                    psf_inten,
                    [ks // 2, ks // 2, ks // 2, ks // 2],
                    mode="constant",
                    value=0,
                )
                .squeeze(0)
                .squeeze(0)
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            psf = psf_inten

        # Intensity normalization, shape of [ks, ks] or [h, w]
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        return diff_float(psf)

    def pupil_field(self, points, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True):
        """Compute complex wavefront at exit pupil plane by coherent ray tracing.

        The wavefront is flipped for subsequent PSF calculation and has the same
        size as the image sensor. This function is differentiable.

        Args:
            points (Tensor or list): Single point source position. Shape [3] or [1, 3],
                with x, y in [-1, 1] and z in [-Inf, 0].
            wvln (float, optional): Wavelength in [um]. Defaults to DEFAULT_WAVE.
            spp (int, optional): Number of rays to sample. Must be >= 1,000,000 for
                accurate coherent simulation. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True, center using chief ray. Defaults to True.

        Returns:
            tuple: (wavefront, psf_center) where:
                - wavefront (Tensor): Complex wavefront at exit pupil. Shape [H, H].
                - psf_center (list): Normalized PSF center coordinates [x, y] in [-1, 1].

        Note:
            Default dtype must be torch.float64 for accurate phase calculation.
        """
        assert spp >= 1_000_000, (
            f"Ray sampling {spp} is too small for coherent ray tracing, which may lead to inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        device = self.device

        if isinstance(points, list):
            points = torch.tensor(points, device=device).unsqueeze(0)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 1:
            points = points.unsqueeze(0).to(device)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 2:
            assert points.shape[0] == 1, (
                f"pupil_field only supports single point input, got shape {points.shape}"
            )
        else:
            raise ValueError(f"Unsupported point type {points.type()}.")

        assert points.shape[0] == 1, (
            "Only one point is supported for pupil field calculation."
        )

        # Ray origin in the object space
        scale = self.calc_scale(points[:, 2].item())
        points_obj = points.clone()
        points_obj[:, 0] = points[:, 0] * scale * sensor_w / 2  # x coordinate
        points_obj[:, 1] = points[:, 1] * scale * sensor_h / 2  # y coordinate

        # Ray center determined by chief ray
        # Shape of [N, 2], un-normalized physical coordinates
        if recenter:
            pointc = self.psf_center(points_obj, method="chief_ray")
        else:
            pointc = self.psf_center(points_obj, method="pinhole")

        # Ray-tracing to exit_pupil
        ray = self.sample_from_points(points=points_obj, num_rays=spp, wvln=wvln)
        ray.coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate complex field (same physical size and resolution as the sensor)
        # Complex field is flipped here for further PSF calculation
        pointc_ref = torch.zeros_like(points[:, :2]).to(device)  # [N, 2]
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=self.pixel_size,
            ks=self.sensor_res[1],
            pointc=pointc_ref,
        )
        wavefront = wavefront.squeeze(0)  # [H, H]

        # PSF center (on the sensor plane)
        pointc = pointc[0, :]
        psf_center = [
            pointc[0] / sensor_w * 2,
            pointc[1] / sensor_h * 2,
        ]

        return wavefront, psf_center

    def psf_huygens(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Single wavelength Huygens PSF calculation.

        This function is not differentiable due to its heavy computational cost.

        Steps:
            1, Trace coherent rays to exit-pupil plane.
            2, Treat every ray as a secondary point source emitting a spherical wave.

        Args:
            points (Tensor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            ks (int, optional): Output kernel size.
            wvln (float, optional): Wavelength.
            spp (int, optional): Sample per pixel.
            recenter (bool, optional): Recenter PSF using chief ray.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].

        References:
            [1] "Optical Aberrations Correction in Postprocessing Using Imaging Simulation", TOG 2021

        Note:
            This is different from ZEMAX Huygens PSF, which traces rays to image plane and do plane wave integration.
        """
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device
        wvln_mm = wvln * 1e-3  # Convert wavelength to mm

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        elif len(points.shape) == 2 and points.shape[0] == 1:
            single_point = True
        else:
            raise ValueError(
                f"Points must be of shape [3] or [1, 3], got {points.shape}."
            )

        # Sample rays from object point
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays coherently through the lens to exit pupil
        ray.coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate PSF center (not flipped here)
        if recenter:
            pointc = -self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = -self.psf_center(point_obj, method="pinhole")

        # Build PSF pixel coordinates (sensor plane at z = d_sensor)
        sensor_z = self.d_sensor.item()
        psf_half_size = (ks / 2) * pixel_size  # Physical half-size of PSF region
        x_coords = torch.linspace(
            -psf_half_size + pixel_size / 2,
            psf_half_size - pixel_size / 2,
            ks,
            device=device,
        )
        y_coords = torch.linspace(
            psf_half_size - pixel_size / 2,
            -psf_half_size + pixel_size / 2,
            ks,
            device=device,
        )
        psf_x, psf_y = torch.meshgrid(
            pointc[0, 0] + x_coords, pointc[0, 1] + y_coords, indexing="xy"
        )  # [ks, ks] each

        # Get valid rays only
        valid_mask = ray.is_valid > 0
        valid_pos = ray.o[valid_mask]  # [num_valid, 3]
        valid_dir = ray.d[valid_mask]  # [num_valid, 3]
        valid_opl = ray.opl[valid_mask]  # [num_valid]
        num_valid = valid_pos.shape[0]

        # Huygens integration: sum spherical waves from each secondary source
        psf_complex = torch.zeros(ks, ks, dtype=torch.complex128, device=device)
        opl_min = valid_opl.min()

        # Compute distance from each secondary source to each pixel
        batch_size = min(num_valid, 10_000)  # Process rays in batches
        for batch_start in range(0, num_valid, batch_size):
            batch_end = min(batch_start + batch_size, num_valid)

            # Batch ray data
            batch_pos = valid_pos[batch_start:batch_end]  # [batch, 3]
            batch_dir = valid_dir[batch_start:batch_end]  # [batch, 3]
            batch_opl = valid_opl[batch_start:batch_end].squeeze(-1)  # [batch]

            # Distance from each secondary source to each pixel
            # batch_pos: [batch, 3], psf_x: [ks, ks]
            dx = psf_x.unsqueeze(-1) - batch_pos[:, 0]  # [ks, ks, batch]
            dy = psf_y.unsqueeze(-1) - batch_pos[:, 1]  # [ks, ks, batch]
            dz = sensor_z - batch_pos[:, 2]  # [batch]

            # Distance r from secondary source to pixel
            r = torch.sqrt(dx**2 + dy**2 + dz**2)  # [ks, ks, batch]

            # Obliquity factor: cos(theta) where theta is angle from normal
            # Using ray direction at exit pupil (dz component)
            obliq = torch.abs(batch_dir[:, 2])  # [batch]
            amp = 0.5 * (1.0 + obliq)  # Huygens–Fresnel obliquity factor

            # Total optical path = OPL through lens + distance to pixel
            total_opl = batch_opl + r  # [ks, ks, batch]

            # Phase relative to reference
            phase = torch.fmod((total_opl - opl_min) / wvln_mm, 1.0) * (
                2 * torch.pi
            )  # [ks, ks, batch]

            # Complex amplitude: A * exp(i * phase) / r (spherical wave decay)
            # We use 1/r for spherical wave amplitude decay
            complex_amp = (amp / r) * torch.exp(1j * phase)  # [ks, ks, batch]

            # Sum contributions from this batch
            psf_complex += complex_amp.sum(dim=-1)  # [ks, ks]

        # Convert complex field to intensity
        psf = psf_complex.abs() ** 2

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        # Flip PSF
        psf = torch.flip(psf, [-2, -1])

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_map(
        self,
        depth=DEPTH,
        grid=(7, 7),
        ks=PSF_KS,
        spp=SPP_PSF,
        wvln=DEFAULT_WAVE,
        recenter=True,
    ):
        """Compute the geometric PSF map at given depth.

        Overrides the base method in Lens class to improve efficiency by parallel ray tracing over different field points.

        Args:
            depth (float, optional): Depth of the object plane. Defaults to DEPTH.
            grid (int, tuple): Grid size (grid_w, grid_h). Defaults to 7.
            ks (int, optional): Kernel size. Defaults to PSF_KS.
            spp (int, optional): Sample per pixel. Defaults to SPP_PSF.
            recenter (bool, optional): Recenter PSF using chief ray. Defaults to True.

        Returns:
            psf_map: PSF map. Shape of [grid_h, grid_w, 1, ks, ks].
        """
        if isinstance(grid, int):
            grid = (grid, grid)
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # [grid_h * grid_w, 1, ks, ks]

        psf_map = psfs.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    @torch.no_grad()
    def psf_center(self, points, method="chief_ray"):
        """Compute reference PSF center (flipped to match the original point) for given point source.

        Args:
            points: [..., 3] un-normalized point is in object plane. [-Inf, Inf] * [-Inf, Inf] * [-Inf, 0]

        Returns:
            psf_center: [..., 2] un-normalized psf center in sensor plane.
        """
        if method == "chief_ray":
            # Shrink the pupil and calculate green light centroid ray as the chief ray
            ray = self.sample_from_points(points, scale_pupil=0.5, num_rays=SPP_CALC)
            ray = self.trace2sensor(ray)
            if not (ray.is_valid == 1).any():
                raise RuntimeError(
                    "When tracing chief ray for PSF center calculation, no ray arrives at the sensor."
                )
            psf_center = ray.centroid()
            psf_center = -psf_center[..., :2]  # shape [..., 2]

        elif method == "pinhole":
            # Pinhole camera perspective projection, distortion not considered
            if points[..., 2].min().abs() < 100:
                print(
                    "Point source is too close, pinhole model may be inaccurate for PSF center calculation."
                )
            tan_point_fov_x = -points[..., 0] / points[..., 2]
            tan_point_fov_y = -points[..., 1] / points[..., 2]
            psf_center_x = self.foclen * tan_point_fov_x
            psf_center_y = self.foclen * tan_point_fov_y
            psf_center = torch.stack([psf_center_x, psf_center_y], dim=-1).to(
                self.device
            )

        else:
            raise ValueError(
                f"Unsupported method for PSF center calculation: {method}."
            )

        return psf_center

    # ====================================================================================
    # Classical optical design
    # ====================================================================================
    def analysis_spot(self, num_field=3, depth=float("inf")):
        """Compute sensor plane ray spot RMS error and radius.

        Analyzes spot sizes across the field of view for multiple wavelengths
        (red, green, blue) and reports statistics.

        Args:
            num_field (int, optional): Number of field positions to analyze along the
                radial direction. Defaults to 3.
            depth (float, optional): Depth of the point source. Use float('inf') for
                collimated light. Defaults to float('inf').

        Returns:
            dict: Spot analysis results keyed by field position (e.g., 'fov0.0', 'fov0.5').
                Each entry contains 'rms' (RMS radius in um) and 'radius' (geometric radius in um).
        """
        rms_radius_fields = []
        geo_radius_fields = []
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Sample rays along meridional (y) direction, shape [num_field, num_rays, 3]
            ray = self.sample_radial_rays(
                num_field=num_field, depth=depth, num_rays=SPP_PSF, wvln=wvln
            )
            ray = self.trace2sensor(ray)

            # Green light point center for reference, shape [num_field, 1, 2]
            if i == 0:
                ray_xy_center_green = ray.centroid()[..., :2].unsqueeze(-2)

            # Calculate RMS spot size and radius for different FoVs
            ray_xy_norm = (
                ray.o[..., :2] - ray_xy_center_green
            ) * ray.is_valid.unsqueeze(-1)
            spot_rms = ((ray_xy_norm**2).sum(-1).sqrt() * ray.is_valid).sum(-1) / (
                ray.is_valid.sum(-1) + EPSILON
            )
            spot_radius = (ray_xy_norm**2).sum(-1).sqrt().max(dim=-1).values

            # Append to list
            rms_radius_fields.append(spot_rms)
            geo_radius_fields.append(spot_radius)

        # Average over wavelengths, shape [num_field]
        avg_rms_radius_um = torch.stack(rms_radius_fields, dim=0).mean(dim=0) * 1000.0
        avg_geo_radius_um = torch.stack(geo_radius_fields, dim=0).mean(dim=0) * 1000.0

        # Print results
        print(f"Ray spot analysis results for depth {depth}:")
        print(
            f"RMS radius: FoV (0.0) {avg_rms_radius_um[0]:.3f} um, FoV (0.5) {avg_rms_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_rms_radius_um[-1]:.3f} um"
        )
        print(
            f"Geo radius: FoV (0.0) {avg_geo_radius_um[0]:.3f} um, FoV (0.5) {avg_geo_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_geo_radius_um[-1]:.3f} um"
        )

        # Save to dict
        rms_results = {}
        fov_ls = torch.linspace(0, 1, num_field)
        for i in range(num_field):
            fov = round(fov_ls[i].item(), 2)
            rms_results[f"fov{fov}"] = {
                "rms": round(avg_rms_radius_um[i].item(), 4),
                "radius": round(avg_geo_radius_um[i].item(), 4),
            }

        return rms_results

    # ====================================================================================
    # Geometrical optics calculation
    # ====================================================================================

    def find_diff_surf(self):
        """Get differentiable/optimizable surface indices.

        Returns a list of surface indices that can be optimized during lens design.
        Excludes the aperture surface from optimization.

        Returns:
            list or range: Surface indices excluding the aperture.
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )
        return diff_surf_range

    @torch.no_grad()
    def calc_foclen(self):
        """Compute effective focal length (EFL).

        Traces a paraxial chief ray and computes the image height, then uses the image height to compute the EFL.

        Updates:
            self.efl: Effective focal length.
            self.foclen: Alias for effective focal length.
            self.bfl: Back focal length (distance from last surface to sensor).

        Reference:
            [1] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Tutorial_MorelSophie.pdf
            [2] https://rafcamera.com/info/imaging-theory/back-focal-length
        """
        # Trace a paraxial chief ray, shape [1, 1, num_rays, 3]
        paraxial_fov = 0.01
        paraxial_fov_deg = float(np.rad2deg(paraxial_fov))
        ray = self.sample_parallel(
            fov_x=0.0, fov_y=paraxial_fov_deg, entrance_pupil=False, scale_pupil=0.2
        )
        ray = self.trace2sensor(ray)

        # Compute the effective focal length
        paraxial_imgh = (ray.o[:, 1] * ray.is_valid).sum() / ray.is_valid.sum()
        eff_foclen = paraxial_imgh.item() / float(np.tan(paraxial_fov))
        self.efl = eff_foclen
        self.foclen = eff_foclen

        # Compute the back focal length
        self.bfl = self.d_sensor.item() - self.surfaces[-1].d.item()

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
        breakpoint()
        return n * math.sin(math.atan(1 / 2 / self.fnum))
        # return n / (2 * self.fnum)

    @torch.no_grad()
    def calc_focal_plane(self, wvln=DEFAULT_WAVE):
        """Compute the focus distance in the object space. Ray starts from sensor center and traces to the object space.

        Args:
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.

        Returns:
            focal_plane (float): Focal plane in the object space.
        """
        device = self.device

        # Sample point source rays from sensor center
        o1 = torch.tensor([0, 0, self.d_sensor.item()]).repeat(SPP_CALC, 1)
        o1 = o1.to(device)

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
            d_sensor (torch.Tensor): Sensor plane in the image space.
        """
        # Sample and trace rays, shape [SPP_CALC, 3]
        if depth == float("inf"):
            ray = self.sample_parallel(
                fov_x=0.0, fov_y=0.0, num_rays=SPP_CALC, wvln=DEFAULT_WAVE
            )
        else:
            ray = self.sample_from_points(
                points=torch.tensor([0.0, 0.0, depth]),
                num_rays=SPP_CALC,
                wvln=DEFAULT_WAVE,
            )
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
        """Compute FoV (radian) of the lens.

        We implement two types of FoV calculation:
            1. Perspective projection from focal length and sensor size.
            2. Ray tracing to compute output ray angle.

        Reference:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        # 1. Perspective projection (effective FoV)
        self.vfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.hfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov = self.dfov / 2  # radius (half diagonal) FoV

        # 2. Ray tracing to calculate real FoV (distortion-affected FoV)
        # Sample rays from edge of sensor, shape [SPP_CALC, 3]
        o1 = torch.zeros([SPP_CALC, 3])
        o1 = torch.tensor([self.r_sensor, 0, self.d_sensor.item()]).repeat(SPP_CALC, 1)

        # Sample second points on exit pupil
        pupilz, pupilx = self.get_exit_pupil()
        x2 = torch.linspace(-pupilx, pupilx, SPP_CALC)
        z2 = torch.full_like(x2, pupilz)
        y2 = torch.full_like(x2, 0)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        # Ray tracing to object space
        ray = Ray(o1, o2 - o1, device=self.device)
        ray = self.trace2obj(ray)

        # Compute output ray angle
        tan_rfov = ray.d[..., 0] / ray.d[..., 2]
        rfov = torch.atan(torch.sum(tan_rfov * ray.is_valid) / torch.sum(ray.is_valid))

        # If calculation failed, use pinhole camera model to compute fov
        if torch.isnan(rfov):
            self.real_rfov = self.rfov
            self.real_dfov = self.dfov
            print(
                f"Failed to calculate distorted FoV by ray tracing, use effective FoV {self.rfov} rad."
            )
        else:
            self.real_rfov = rfov.item()
            self.real_dfov = 2 * rfov.item()

        # 3. Compute 35mm equivalent focal length. 35mm sensor: 36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov)

    @torch.no_grad()
    def calc_scale(self, depth):
        """Calculate the scale factor (object height / image height).

        Uses the pinhole camera model to compute magnification.

        Args:
            depth (float): Object distance from the lens (negative z direction).

        Returns:
            float: Scale factor relating object height to image height.
        """
        return -depth / self.foclen

    @torch.no_grad()
    def calc_pupil(self):
        """Compute entrance and exit pupil positions and radii.

        The entrance and exit pupils must be recalculated whenever:
            - First-order parameters change (e.g., field of view, object height, image height),
            - Lens geometry or materials change (e.g., surface curvatures, refractive indices, thicknesses),
            - Or generally, any time the lens configuration is modified.

        Updates:
            self.aper_idx: Index of the aperture surface.
            self.exit_pupilz, self.exit_pupilr: Exit pupil position and radius.
            self.entr_pupilz, self.entr_pupilr: Entrance pupil position and radius.
            self.exit_pupilz_parax, self.exit_pupilr_parax: Paraxial exit pupil.
            self.entr_pupilz_parax, self.entr_pupilr_parax: Paraxial entrance pupil.
            self.fnum: F-number calculated from focal length and entrance pupil.
        """
        # Find aperture
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Aperture):
                self.aper_idx = i
                break

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])
            print("No aperture found, use the smallest surface as aperture.")

        # Compute entrance and exit pupil
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil(paraxial=False)
        self.entr_pupilz, self.entr_pupilr = self.calc_entrance_pupil(paraxial=False)
        self.exit_pupilz_parax, self.exit_pupilr_parax = self.calc_exit_pupil(
            paraxial=True
        )
        self.entr_pupilz_parax, self.entr_pupilr_parax = self.calc_entrance_pupil(
            paraxial=True
        )

        # Compute F-number
        self.fnum = self.foclen / (2 * self.entr_pupilr)

    def get_entrance_pupil(self, paraxial=False):
        """Get entrance pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            tuple: (z_position, radius) of the entrance pupil in [mm].
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
            tuple: (z_position, radius) of the exit pupil in [mm].
        """
        if paraxial:
            return self.exit_pupilz_parax, self.exit_pupilr_parax
        else:
            return self.exit_pupilz, self.exit_pupilr

    @torch.no_grad()
    def calc_exit_pupil(self, paraxial=False):
        """Calculate exit pupil location and radius.

        Paraxial mode:
            Rays are emitted from near the center of the aperture stop and are close to the optical axis.
            This mode estimates the exit pupil position and radius under ideal (first-order) optical assumptions.
            It is fast and stable.

        Non-paraxial mode:
            Rays are emitted from the edge of the aperture stop in large quantities.
            The exit pupil position and radius are determined based on the intersection points of these rays.
            This mode is slower and affected by aperture-related aberrations.

        Use paraxial mode unless precise ray aiming is required.

        Args:
            paraxial (bool): center (True) or edge (False).

        Returns:
            avg_pupilz (float): z coordinate of exit pupil.
            avg_pupilr (float): radius of exit pupil.

        Reference:
            [1] Exit pupil: how many rays can come from sensor to object space.
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            print("No aperture, use the last surface as exit pupil.")
            return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # Sample rays from aperture (edge or center)
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]]).repeat(32, 1)
            phi_rad = torch.linspace(-0.01, 0.01, 32)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi_rad = torch.linspace(-rfov / 2, rfov / 2, SPP_CALC)

        d = torch.stack(
            (torch.sin(phi_rad), torch.zeros_like(phi_rad), torch.cos(phi_rad)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

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
            avg_pupilz = self.surfaces[-1].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative exit pupil is detected, use the last surface as pupil."
                )
                avg_pupilr = self.surfaces[-1].r
                avg_pupilz = self.surfaces[-1].d.item()

        return avg_pupilz, avg_pupilr

    @torch.no_grad()
    def calc_entrance_pupil(self, paraxial=False):
        """Calculate entrance pupil of the lens.

        The entrance pupil is the optical image of the physical aperture stop, as seen through the optical elements in front of the stop. We sample backward rays from the aperture stop and trace them to the first surface, then find the intersection points of the reverse extension of the rays. The average of the intersection points defines the entrance pupil position and radius.

        Args:
            paraxial (bool): Ray sampling mode. Default: True.
                - True: Rays emitted from near the center of the aperture stop, close to
                  the optical axis. Fast and stable under ideal optical assumptions.
                - False: Rays emitted from the edge of the aperture stop in large quantities.
                  Slower and affected by aperture-related aberrations.

        Returns:
            tuple: (z_position, radius) of entrance pupil.

        Note:
            [1] Use paraxial mode unless precise ray aiming is required.
            [2] This function only works for object at a far distance. For microscopes, this function usually returns a negative entrance pupil.

        References:
            [1] Entrance pupil: how many rays can come from object space to sensor.
            [2] https://en.wikipedia.org/wiki/Entrance_pupil: "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
            [3] Zemax LLC, *OpticStudio User Manual*, Version 19.4, Document No. 2311, 2019.
        """
        if self.aper_idx is None or not hasattr(self, "aper_idx"):
            print("No aperture stop, use the first surface as entrance pupil.")
            return self.surfaces[0].d.item(), self.surfaces[0].r

        # Sample rays from edge of aperture stop
        aper_idx = self.aper_idx
        aper_surf = self.surfaces[aper_idx]
        aper_z = aper_surf.d.item()
        if aper_surf.is_square:
            aper_r = float(np.sqrt(2)) * aper_surf.r
        else:
            aper_r = aper_surf.r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]]).repeat(32, 1)
            phi = torch.linspace(-0.01, 0.01, 32)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi = torch.linspace(-rfov / 2, rfov / 2, SPP_CALC)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

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
            avg_pupilz = self.surfaces[0].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface as entrance pupil."
                )
                avg_pupilr = self.surfaces[0].r
                avg_pupilz = self.surfaces[0].d.item()

        return avg_pupilz, avg_pupilr

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """Compute the intersection points of 2D lines.

        Args:
            origins (torch.Tensor): Origins of the lines. Shape: [N, 2]
            directions (torch.Tensor): Directions of the lines. Shape: [N, 2]

        Returns:
            torch.Tensor: Intersection points. Shape: [N*(N-1)/2, 2]
        """
        N = origins.shape[0]

        # Create pairwise combinations of indices
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # Shape: [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # Shape: [N*(N-1)/2, 2]
        Di = directions[idx_i]  # Shape: [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # Shape: [N*(N-1)/2, 2]

        # Vector from Oi to Oj
        b = Oj - Oi  # Shape: [N*(N-1)/2, 2]

        # Coefficients matrix A
        A = torch.stack([Di, -Dj], dim=-1)  # Shape: [N*(N-1)/2, 2, 2]

        # Solve the linear system Ax = b
        # Using least squares to handle the case of no exact solution
        if A.device.type == "mps":
            # Perform lstsq on CPU for MPS devices and move result back
            x, _ = torch.linalg.lstsq(A.cpu(), b.unsqueeze(-1).cpu())[:2]
            x = x.to(A.device)
        else:
            x, _ = torch.linalg.lstsq(A, b.unsqueeze(-1))[:2]
        x = x.squeeze(-1)  # Shape: [N*(N-1)/2, 2]
        s = x[:, 0]
        t = x[:, 1]

        # Calculate the intersection points using either rays
        P_i = Oi + s.unsqueeze(-1) * Di  # Shape: [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # Shape: [N*(N-1)/2, 2]

        # Take the average to mitigate numerical precision issues
        P = (P_i + P_j) / 2

        return P

    # ====================================================================================
    # Lens operation
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, foc_dist=float("inf")):
        """Refocus the lens to a depth distance by changing sensor position.

        Args:
            foc_dist (float): focal distance.

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
        """Set F-number and aperture radius using binary search.

        Args:
            fnum (float): target F-number.
        """
        current_fnum = self.fnum
        current_aper_r = self.surfaces[self.aper_idx].r
        target_pupil_r = self.foclen / fnum / 2

        # Binary search to find aperture radius that gives desired exit pupil radius
        aper_r = current_aper_r * (current_fnum / fnum)
        aper_r_min = 0.5 * aper_r
        aper_r_max = 2.0 * aper_r

        for _ in range(16):
            self.surfaces[self.aper_idx].r = aper_r
            _, pupilr = self.calc_entrance_pupil()

            if abs(pupilr - target_pupil_r) < 0.1:  # Close enough
                break

            if pupilr > target_pupil_r:
                # Current radius is too large, decrease it
                aper_r_max = aper_r
                aper_r = (aper_r_min + aper_r) / 2
            else:
                # Current radius is too small, increase it
                aper_r_min = aper_r
                aper_r = (aper_r_max + aper_r) / 2

        self.surfaces[self.aper_idx].r = aper_r

        # Update pupil after setting aperture radius
        self.calc_pupil()

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov, fnum):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        Args:
            rfov (float): half diagonal-FoV in radian.
            fnum (float): F number.
        """
        if rfov > math.pi:
            self.rfov = rfov / 180.0 * math.pi
        else:
            self.rfov = rfov

        self.foclen = self.r_sensor / math.tan(self.rfov)
        self.fnum = fnum
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].update_r(float(aper_r))

        # Update pupil after setting aperture radius
        self.calc_pupil()

    @torch.no_grad()
    def set_fov(self, rfov):
        """Set FoV. This function is used to assign design targets.

        Args:
            rfov (float): half diagonal-FoV in radian.
        """
        self.rfov = rfov

    @torch.no_grad()
    def prune_surf(self, expand_factor=None):
        """Prune surfaces to allow all valid rays to go through.

        Args:
            expand_factor (float): height expansion factor.
                - For cellphone lens, we usually expand by 5%
                - For camera lens, we usually expand by 20%.
        """
        surface_range = self.find_diff_surf()

        # Set expansion factor
        if self.r_sensor < 10.0:
            expand_factor = 0.05 if expand_factor is None else expand_factor
        else:
            expand_factor = 0.10 if expand_factor is None else expand_factor

        # Expand surface height
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].r * (1 + expand_factor)

        # Sample and trace rays to compute the maximum valid region
        if self.rfov is not None:
            fov_deg = self.rfov * 180 / torch.pi
        else:
            fov = np.arctan(self.r_sensor / self.foclen)
            fov_deg = float(fov) * 180 / torch.pi
            print(f"Using fov_deg: {fov_deg} during surface pruning.")

        fov_y = [f * fov_deg / 10 for f in range(0, 11)]
        ray = self.sample_parallel(
            fov_x=[0.0], fov_y=fov_y, num_rays=SPP_CALC, scale_pupil=1.5
        )
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(ray_o_record, 0.0)
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # Update surface height
        for i in surface_range:
            if surf_r_max[i] > 0:
                r_expand = surf_r_max[i].item() * expand_factor
                r_expand = max(min(r_expand, 2.0), 0.1)
                self.surfaces[i].update_r(surf_r_max[i].item() + r_expand)
            else:
                print(f"No valid rays for Surf {i}, expand existing radius.")
                r_expand = self.surfaces[i].r * expand_factor
                r_expand = max(min(r_expand, 2.0), 0.1)
                self.surfaces[i].update_r(self.surfaces[i].r + r_expand)

    @torch.no_grad()
    def correct_shape(self, expand_factor=None):
        """Correct wrong lens shape during lens design optimization.

        Applies correction rules to ensure valid lens geometry:
            1. Move the first surface to z = 0.0
            2. Fix aperture distance if aperture is at the front
            3. Prune all surfaces to allow valid rays through

        Args:
            expand_factor (float, optional): Height expansion factor for surface pruning.
                If None, auto-selects based on lens type. Defaults to None.

        Returns:
            bool: True if any shape corrections were made, False otherwise.
        """
        aper_idx = self.aper_idx
        optim_surf_range = self.find_diff_surf()
        shape_changed = False

        # Rule 1: Move the first surface to z = 0.0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # Rule 2: Fix aperture distance to the first surface if aperture in the front.
        if aper_idx == 0:
            d_aper = 0.05

            # If the first surface is concave, use the maximum negative sag.
            aper_r = torch.tensor(self.surfaces[aper_idx].r, device=self.device)
            sag1 = -self.surfaces[aper_idx + 1].sag(aper_r, 0).item()

            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in optim_surf_range:
                self.surfaces[i].d -= delta_aper
            self.d_sensor -= delta_aper

        # Rule 4: Prune all surfaces
        self.prune_surf(expand_factor=expand_factor)

        if shape_changed:
            print("Surface shape corrected.")
        return shape_changed

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match lens materials to a glass catalog.

        Args:
            mat_table (str, optional): Glass catalog name. Common options include
                'CDGM', 'SCHOTT', 'OHARA'. Defaults to 'CDGM'.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)

    # ====================================================================================
    # Visualization and numerical analysis
    # ====================================================================================
    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        depth=float("inf"),
        render=False,
        render_unwarp=False,
        lens_title=None,
        show=False,
    ):
        """Analyze the optical lens.

        Args:
            save_name (str): save name.
            depth (float): object depth distance.
            render (bool): whether render an image.
            render_unwarp (bool): whether unwarp the rendered image.
            lens_title (str): lens title
            show (bool): whether to show the rendered image.
        """
        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
            show=show,
        )

        # Draw spot diagram
        self.draw_spot_radial(
            save_name=f"{save_name}_spot.png",
            depth=depth,
            show=show,
        )

        # Draw MTF
        if depth == float("inf"):
            # This is a hack to draw MTF for infinite depth
            self.draw_mtf(
                depth_list=[DEPTH], save_name=f"{save_name}_mtf.png", show=show
            )
        else:
            self.draw_mtf(
                depth_list=[depth], save_name=f"{save_name}_mtf.png", show=show
            )

        # Calculate RMS error
        self.analysis_spot(depth=depth)

        # Render an image, compute PSNR and SSIM
        if render:
            depth = DEPTH if depth == float("inf") else depth
            img_org = Image.open("./datasets/charts/NBS_1963_1k.png").convert("RGB")
            img_org = np.array(img_org)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                noise=0.01,
                show=show,
            )

    # ====================================================================================
    # Optimization
    # ====================================================================================
    def get_optimizer_params(
        self,
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        decay=0.01,
        optim_mat=False,
        optim_surf_range=None,
    ):
        """Get optimizer parameters for different lens surface.

        Recommendation:
            For cellphone lens: [d, c, k, a], [1e-4, 1e-4, 1e-1, 1e-4]
            For camera lens: [d, c, 0, 0], [1e-3, 1e-4, 0, 0]

        Args:
            lrs (list): learning rate for different parameters.
            decay (float): decay rate for higher order a. Defaults to 0.01.
            optim_mat (bool): whether to optimize material. Defaults to False.
            optim_surf_range (list): surface indices to be optimized. Defaults to None.

        Returns:
            list: optimizer parameters
        """
        # Find surfaces to be optimized
        if optim_surf_range is None:
            # optim_surf_range = self.find_diff_surf()
            optim_surf_range = range(len(self.surfaces))

        # If lr for each surface is a list is given
        if isinstance(lrs[0], list):
            return self.get_optimizer_params_manual(
                lrs=lrs, optim_mat=optim_mat, optim_surf_range=optim_surf_range
            )

        # Optimize lens surface parameters
        params = []
        for surf_idx in optim_surf_range:
            surf = self.surfaces[surf_idx]

            if isinstance(surf, Aperture):
                params += surf.get_optimizer_params(lrs=[lrs[0]])

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(
                    lrs=lrs[:4], decay=decay, optim_mat=optim_mat
                )

            elif isinstance(surf, AsphericNorm):
                params += surf.get_optimizer_params(
                    lrs=lrs[:4], decay=decay, optim_mat=optim_mat
                )

            elif isinstance(surf, Phase):
                params += surf.get_optimizer_params(lrs=[lrs[0], lrs[4]])

            # elif isinstance(surf, GaussianRBF):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            # elif isinstance(surf, NURBS):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Plane):
                params += surf.get_optimizer_params(lrs=[lrs[0]], optim_mat=optim_mat)

            # elif isinstance(surf, PolyEven):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            elif isinstance(surf, ThinLens):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            else:
                raise Exception(
                    f"Surface type {surf.__class__.__name__} is not supported for optimization yet."
                )

        # Optimize sensor place
        self.d_sensor.requires_grad = True
        params += [{"params": self.d_sensor, "lr": lrs[0]}]

        return params

    def get_optimizer(
        self,
        lrs=[1e-4, 1e-4, 1e-1, 1e-4],
        decay=0.01,
        optim_surf_range=None,
        optim_mat=False,
    ):
        """Get optimizers and schedulers for different lens parameters.

        Args:
            lrs (list): learning rate for different parameters [c, d, k, a]. Defaults to [1e-4, 1e-4, 0, 1e-4].
            decay (float): decay rate for higher order a. Defaults to 0.2.
            optim_surf_range (list): surface indices to be optimized. Defaults to None.
            optim_mat (bool): whether to optimize material. Defaults to False.

        Returns:
            list: optimizer parameters
        """
        # Initialize lens design constraints (edge thickness, etc.)
        self.init_constraints()

        # Get optimizer
        params = self.get_optimizer_params(
            lrs=lrs, decay=decay, optim_surf_range=optim_surf_range, optim_mat=optim_mat
        )
        optimizer = torch.optim.Adam(params)
        # optimizer = torch.optim.SGD(params)
        return optimizer

    # ====================================================================================
    # Lens file IO
    # ====================================================================================
    def read_lens_json(self, filename="./test.json"):
        """Read the lens from a JSON file.

        Loads lens configuration including surfaces, materials, and optical properties
        from the DeepLens native JSON format.

        Args:
            filename (str, optional): Path to the JSON lens file. Defaults to './test.json'.

        Note:
            After loading, the lens is moved to self.device and post_computation is called
            to calculate derived properties.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, "r") as f:
            data = json.load(f)
            d = 0.0
            for idx, surf_dict in enumerate(data["surfaces"]):
                surf_dict["d"] = d
                surf_dict["surf_idx"] = idx

                if surf_dict["type"] == "Aperture":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Aspheric":
                    # s = Aspheric.init_from_dict(surf_dict)
                    s = AsphericNorm.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Cubic":
                    s = Cubic.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "GaussianRBF":
                #     s = GaussianRBF.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "NURBS":
                #     s = NURBS.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Phase":
                    s = Phase.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Plane":
                    s = Plane.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "PolyEven":
                #     s = PolyEven.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Stop":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Spheric":
                    s = Spheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "ThinLens":
                    s = ThinLens.init_from_dict(surf_dict)

                else:
                    raise Exception(
                        f"Surface type {surf_dict['type']} is not implemented in GeoLens.read_lens_json()."
                    )

                self.surfaces.append(s)
                d += surf_dict["d_next"]

        self.d_sensor = torch.tensor(d)
        self.lens_info = data.get("info", "None")
        self.enpd = data.get("enpd", None)
        self.float_enpd = True if self.enpd is None else False
        self.float_foclen = False
        self.float_rfov = False
        self.r_sensor = data["r_sensor"]

        self.to(self.device)

        # Set sensor size and resolution
        sensor_res = data.get("sensor_res", (2000, 2000))
        self.set_sensor_res(sensor_res=sensor_res)
        self.post_computation()

    def write_lens_json(self, filename="./test.json"):
        """Write the lens to a JSON file.

        Saves the complete lens configuration including all surfaces, materials,
        focal length, F-number, and sensor properties to the DeepLens JSON format.

        Args:
            filename (str, optional): Path for the output JSON file. Defaults to './test.json'.
        """
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["foclen"] = round(self.foclen, 4)
        data["fnum"] = round(self.fnum, 4)
        if self.float_enpd is False:
            data["enpd"] = round(self.enpd, 4)
        data["r_sensor"] = self.r_sensor
        data["(d_sensor)"] = round(self.d_sensor.item(), 4)
        data["(sensor_size)"] = [round(i, 4) for i in self.sensor_size]
        data["surfaces"] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i}
            surf_dict.update(s.surf_dict())
            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = round(
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item(), 4
                )
            else:
                surf_dict["d_next"] = round(
                    self.d_sensor.item() - self.surfaces[i].d.item(), 4
                )

            data["surfaces"].append(surf_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Lens written to {filename}")

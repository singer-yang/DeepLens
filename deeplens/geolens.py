# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""A geometric lens consisting of refractive surfaces, simulate with ray tracing. May contain diffractive surfaces, but still use ray tracing to simulate.

For image simulation:
    1. Ray tracing based rendering
    2. PSF + patch convolution

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import json
import math

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from deeplens.geolens_pkg.eval import GeoLensEval
from deeplens.geolens_pkg.io import GeoLensIO
from deeplens.geolens_pkg.optim import GeoLensOptim
from deeplens.geolens_pkg.vis import GeoLensVis
from deeplens.lens import Lens
from deeplens.optics.basics import (
    DEFAULT_WAVE,
    DEPTH,
    DELTA,
    DELTA_PARAXIAL,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
    SPP_RENDER,
    SPP_PARAXIAL,
    WAVE_RGB,
)
from deeplens.optics.geometric_surface import (
    Aperture,
    Aspheric,
    AsphericNorm,
    Cubic,
    Phase,
    Plane,
    Spheric,
    ThinLens,
)
from deeplens.optics.materials import Material
from deeplens.optics.monte_carlo import forward_integral
from deeplens.optics.ray import Ray
from deeplens.optics.utils import diff_float
from deeplens.optics.wave import AngularSpectrumMethod
from deeplens.utils import (
    batch_psnr,
    batch_ssim,
    denormalize_ImageNet,
    img2batch,
    normalize_ImageNet,
)


class GeoLens(Lens, GeoLensEval, GeoLensOptim, GeoLensVis, GeoLensIO):
    def __init__(
        self,
        filename=None,
        sensor_res=(1000, 1000),
        sensor_size=(8.0, 8.0),
        device=None,
    ):
        """Initialize a refractive lens.

        There are three ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces and materials
        """
        super().__init__(device)

        # Lens sensor size and resolution
        self.sensor_res = sensor_res
        self.sensor_size = sensor_size
        self.r_sensor = float(np.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2)) / 2

        # Load lens file
        if filename is not None:
            self.read_lens(filename)
        else:
            self.surfaces = []
            self.materials = []
            self.to(self.device)

        # Initialize lens design constraints (edge thickness, etc.)
        self.init_constraints()

    def read_lens(self, filename):
        """Read a GeoLens from a file.

        In this step, sensor size and resolution will usually be overwritten.
        """
        # Load lens file
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_zmx(filename)
        elif filename[-4:] == ".seq":
            raise NotImplementedError("File format .seq is not supported yet.")
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # After loading lens, compute foclen, fov and fnum
        self.to(self.device)
        self.post_computation()

    def post_computation(self):
        """After loading lens, compute foclen, fov and fnum."""
        # Basic lens parameter calculation
        self.calc_pupil(paraxial=False)
        self.foclen = self.calc_efl()
        self.hfov = self.calc_hfov()
        self.fnum = self.calc_fnum()

    def update_float_setting(self):
        """After lens changed, compute foclen, fov and fnum."""
        # Basic lens parameter calculation
        self.calc_pupil(paraxial=False)
        if self.float_enpd is False:
            self.entrance_pupilr = self.enpd / 2.0
        if self.float_foclen is True:
            self.foclen = self.calc_efl()
        if self.float_hfov is True:
            self.hfov = self.calc_hfov()
        self.fnum = self.calc_fnum()

    def double(self):
        """Use double-precision for coherent ray tracing."""
        torch.set_default_dtype(torch.float64)
        for surf in self.surfaces:
            surf.double()

    def __call__(self, ray):
        """The input and output of a GeoLens object are both Ray objects."""
        return self.trace(ray)

    # ====================================================================================
    # Ray sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_grid_rays(
        self,
        num_grid=[11, 11],
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """Sample grid rays from object space.
            (1) If depth is infinite, sample parallel rays at different field angles.
            (2) If depth is finite, sample point source rays from the object plane.

        This function is usually used for (1) PSF map, (2) RMS error map, and (3) spot diagram calculation.

        Args:
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_grid (list, optional): number of grid points. Defaults to [11, 11].
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            sample_more_off_axis (bool, optional): If True, sample more off-axis rays.
            scale_pupil (float, optional): Scale factor for pupil radius.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid, num_grid, num_rays, 3]
        """
        if isinstance(num_grid, int):
            num_grid = [num_grid, num_grid]

        # Calculate field angles for grid source. Top-left field has positive fov_x and negative fov_y
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # Calculate FoV_x and FoV_y
        hfov_x = np.atan(np.tan(self.hfov) * self.sensor_size[1] / self.r_sensor / 2)
        hfov_y = np.atan(np.tan(self.hfov) * self.sensor_size[0] / self.r_sensor / 2)
        hfov_x = np.rad2deg(hfov_x)
        hfov_y = np.rad2deg(hfov_y)
        fov_x_list = [float(x * hfov_x) for x in x_list]
        fov_y_list = [float(y * hfov_y) for y in y_list]

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
        plane="meridional",
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
        fov_y_list = torch.linspace(
            0, float(np.rad2deg(self.hfov)), num_field, device=self.device
        )

        if depth == float("inf"):
            ray = self.sample_parallel(
                fov_x=[0.0], fov_y=fov_y_list, num_rays=num_rays, wvln=wvln
            )
            ray = ray.squeeze(1)
        else:
            point_obj_x = torch.zeros(num_field, device=self.device)
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
        Sample rays from point sources in object space (absolute 3D coordinates).
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
        pupilz, pupilr = self.get_entrance_pupil(scale_pupil=scale_pupil)
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

        # Calculate rays and propagate to 0.0 (to improve stability)
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
            fov_x (float or list): Field angle(s) in the x–z plane (degrees). Default: [0.0].
            fov_y (float or list): Field angle(s) in the y–z plane (degrees). Default: [0.0].
            num_rays (int): Number of rays per field point. Default: SPP_CALC.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            entrance_pupil (bool): If True, sample origins on entrance pupil; otherwise, on surface 0. Default: True.
            depth (float): Propagation depth in z. Default: -1.0.
            scale_pupil (float): Scale factor for pupil radius. Default: 1.0.

        Returns:
            Ray: Ray object with shape [len(fov_y), len(fov_x), num_rays, 3], ordered as (u, v).
        """
        # Preprocess fov angles
        if isinstance(fov_x, float):
            fov_x = [fov_x]
        if isinstance(fov_y, float):
            fov_y = [fov_y]

        fov_x = torch.tensor([fx * torch.pi / 180 for fx in fov_x]).to(self.device)
        fov_y = torch.tensor([fy * torch.pi / 180 for fy in fov_y]).to(self.device)

        # Sample ray origins on the pupil, shape [num_fov_x, num_fov_y, num_rays, 3]
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil(scale_pupil=scale_pupil)
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        ray_o = self.sample_circle(
            r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
        )

        # Sample ray directions, shape [num_fov_y, num_fov_x, num_rays, 3]
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x, fov_y, indexing="xy")
        dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
        dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
        dz = torch.ones_like(ray_o[..., 2])
        ray_d = torch.stack((dx, dy, dz), dim=-1)

        # Form rays and propagate to the target depth
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
            pupilz, pupilr = self.get_entrance_pupil(scale_pupil=scale_pupil)
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
        """Sample rays from sensor pixels (backward rays). Used for ray-tracing based rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [H, W, spp, 3]
        """
        # Sample points on sensor plane
        # Use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(
                -self.sensor_size[1] / 2,
                self.sensor_size[1] / 2,
                self.sensor_res[1] + 1,
                device=self.device,
            )[1:],
            torch.linspace(
                self.sensor_size[0] / 2,
                -self.sensor_size[0] / 2,
                self.sensor_res[0] + 1,
                device=self.device,
            )[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor.item())

        # Sample second points on the pupil
        pupilz, pupilr = self.calc_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(*self.sensor_res, spp))

        # Form rays
        ray_o = torch.stack((x1, y1, z1), 2)
        ray_o = ray_o.unsqueeze(2).expand(-1, -1, spp, -1)  # [H, W, spp, 3]

        # Sub-pixel sampling for more realistic rendering
        if sub_pixel:
            delta_ox = (
                torch.rand((ray_o[:, :, :, 0].clone().shape), device=self.device)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand((ray_o[:, :, :, 1].clone().shape), device=self.device)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # Form rays
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=self.device)
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

        # Generate random angles
        theta = torch.rand(*shape, device=device) * 2 * torch.pi

        # Generate random radii with square root for uniform distribution
        r2 = torch.rand(*shape, device=device) * r**2
        radius = torch.sqrt(r2 + EPSILON)

        # Convert to Cartesian coordinates
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)

        # Stack to form 3D points
        points = torch.stack((x, y, z_tensor), dim=-1)

        # Fix all chief rays to facilitate the design of telecentricity.
        points[..., 0, :2] = 0.0

        return points

    # ====================================================================================
    # Ray tracing
    # ====================================================================================
    def trace(self, ray, lens_range=None, record=False):
        """Ray tracing function. Forward or backward ray tracing is automatically determined by ray directions.

        Args:
            ray (Ray object): Ray object.
            lens_range (list): lens range.
            record (bool): record ray path or not.

        Returns:
            ray_final (Ray object): ray after optical system.
            ray_o_record (list): list of intersection points.
        """
        # Manually propagate ray to a shallow depth to improve accuracy
        if (ray.o[..., 2].min() < -100.0).any():
            ray = ray.prop_to(-10.0)

        if lens_range is None:
            lens_range = range(0, len(self.surfaces))

        if ray.is_forward.any():
            ray_out, ray_o_record = self.forward_tracing(ray, lens_range, record=record)
        else:
            ray_out, ray_o_record = self.backward_tracing(
                ray, lens_range, record=record
            )

        return ray_out, ray_o_record

    def trace2obj(self, ray, depth=DEPTH):
        """Trace rays through the lens and reach the sensor plane.

        Args:
            ray (Ray object): Ray object.
            depth (float): sensor distance.
        """
        ray, _ = self.trace(ray)
        ray = ray.prop_to(depth)
        return ray

    def trace2sensor(self, ray, record=False):
        """Trace optical rays to sensor plane.

        Args:
            ray (Ray object): Ray object.
            record (bool): record ray path or not.

        Returns:
            ray_out (Ray object): ray after optical system.
            ray_o_record (list): list of intersection points.
        """
        # Manually propagate ray to a shallow depth to improve accuracy
        if (ray.o[..., 2].min() < -100.0).any():
            ray = ray.prop_to(-10.0)

        # Trace rays
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.prop_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            # Set to nan to be skipped in 2d layout visualization
            ray_o[ray.valid == 0] = float("nan")
            ray_o_record.append(ray_o)
            return ray, ray_o_record
        else:
            return ray

    def forward_tracing(self, ray, lens_range, record):
        """Trace rays from object space to sensor plane.

        Args:
            ray (Ray object): Ray object.
            lens_range (list): lens range.
            record (bool): record ray path or not.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in lens_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    def backward_tracing(self, ray, lens_range, record):
        """Trace rays from sensor plane to object space.

        Args:
            ray (Ray object): Ray object.
            lens_range (list): lens range.
            record (bool): record ray path or not.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in np.flip(lens_range):
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i - 1].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i - 1].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    # ====================================================================================
    # Image simulation
    # ====================================================================================
    def render(self, img_obj, depth=DEPTH, method="raytracing", **kwargs):
        """Differentiable image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] PSF patch convolution.
            [3] Ray tracing-based rendering.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. Defaults to "psf".
            **kwargs: Additional arguments for different methods.
        """
        # Check sensor resolution
        if not (
            self.sensor_res[0] == img_obj.shape[-2]
            and self.sensor_res[1] == img_obj.shape[-1]
        ):
            H, W = img_obj.shape[-2], img_obj.shape[-1]
            self.set_sensor(sensor_res=(H, W), sensor_size=self.sensor_size)

        # Differentiable image simulation
        if method == "psf_map":
            # PSF based rendering - uses PSF map to render image
            if "psf_grid" in kwargs and "psf_ks" in kwargs:
                psf_grid, psf_ks = kwargs["psf_grid"], kwargs["psf_ks"]
                img_render = self.render_psf_map(
                    img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
                )
            else:
                # Use default PSF grid and kernel size
                img_render = self.render_psf_map(img_obj, depth=depth)

        elif method == "psf_patch":
            # PSF patch based rendering - uses a single PSF to render a patch of the image
            if "psf_center" in kwargs and "psf_ks" in kwargs:
                psf_center, psf_ks = kwargs["psf_center"], kwargs["psf_ks"]
                img_render, field_channel = self.render_psf_patch(
                    img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
                )
            else:
                img_render = self.render_psf_patch(img_obj, depth=depth)

        elif method == "raytracing":
            # Ray tracing based rendering
            if "spp" in kwargs:
                spp = kwargs["spp"]
                img_render = self.render_raytracing(img_obj, depth=depth, spp=spp)
            else:
                # Use default sample per pixel
                img_render = self.render_raytracing(img_obj, depth=depth)

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_raytracing(self, img, depth=DEPTH, spp=SPP_RENDER, vignetting=False):
        """Render RGB image using ray tracing method.

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
        """Render monochrome image using ray tracing method.

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
        ray = self.trace2obj(ray, depth=depth)
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
        # Preparetion (img is [N, C, H, W] or [N, H, W] tensor)
        if torch.is_tensor(img):
            H, W = img.shape[-2:]
            if len(img.shape) == 4:
                img = F.pad(img, (1, 1, 1, 1), "replicate")
            else:
                img = F.pad(img.unsqueeze(1), (1, 1, 1, 1), "replicate").squeeze(1)
            # img = F.pad(img, (1,1,1,1), "constant")    #constant padding can work for arbitary dmensions
        else:
            raise Exception("Input image should be Tensor.")

        # Scale object image physical size to get 1:1 pixel-pixel alignment with sensor image
        ray = ray.prop_to(depth)
        p = ray.o[..., :2]
        pixel_size = scale * self.pixel_size
        ray.valid = (
            ray.valid
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
            image = torch.sum(irr_img * ray.valid, -1) / (
                torch.sum(ray.valid, -1) + EPSILON
            )
        else:
            image = torch.sum(irr_img * ray.valid, -1) / torch.numel(ray.valid)

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
        method="raytracing",
    ):
        """Render a single image for visualization and analysis. This function is designed to be non-differentiable. If want to use differentiable rendering, call self.render() function.

        Args:
            img_org (tensor): [H, W, 3] shape image.
            depth (float, optional): depth of object image. Defaults to DEPTH.
            spp (int, optional): sample per pixel. Defaults to 64.
            unwarp (bool, optional): unwarp the image. Defaults to False.
            save_name (str, optional): save name. Defaults to None.
            noise (float, optional): sensor noise. Defaults to 0.0.
            method (str, optional): rendering method. Defaults to 'raytracing'.
        """
        # Change sensor resolution to match the image
        sensor_res_original = self.sensor_res
        img = img2batch(img_org).to(self.device)
        self.set_sensor(sensor_res=img.shape[-2:], sensor_size=self.sensor_size)

        # Image rendering
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # Add noise (a very simple Gaussian noise model)
        if noise > 0:
            img_render = img_render + torch.randn_like(img_render) * noise
            img_render = torch.clamp(img_render, 0, 1)

        # Compute PSNR and SSIM
        render_psnr = round(batch_psnr(img, img_render), 3)
        render_ssim = round(batch_ssim(img, img_render), 3)
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.3f}")

        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # Compute PSNR and SSIM
            render_psnr = round(batch_psnr(img, img_render), 3)
            render_ssim = round(batch_ssim(img, img_render), 3)
            print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.3f}")

            if save_name is not None:
                save_image(img_render, f"{save_name}_unwarped.png")

        # Change the sensor resolution back
        self.set_sensor(sensor_res=sensor_res_original, sensor_size=self.sensor_size)

        return img_render

    @torch.no_grad()
    def analysis_end2end(
        self,
        net,
        img_raw=None,
        img_gt=None,
        save_name="./end2end",
        depth=DEPTH,
        render_unwarp=False,
        noise=0.01,
    ):
        """Analysis End2End result with either simulated raw image or captured raw image.

        Args:
            net: image reconstruction network.
        """
        net.eval()
        if img_raw is None:
            img_org = cv.cvtColor(cv.imread("./dataset/0185.png"), cv.COLOR_BGR2RGB)
            img_gt = cv.cvtColor(cv.imread("./dataset/0185.png"), cv.COLOR_BGR2RGB)

            img_raw = self.render_single_img(
                img_org, depth=depth, spp=128, unwarp=render_unwarp, noise=noise
            )

        # Image reconstruction
        img_raw = (
            torch.tensor(img_raw).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        )
        img_rec = denormalize_ImageNet(net(normalize_ImageNet(img_raw)))

        save_image(img_raw, f"{save_name}_raw.png")
        save_image(img_rec, f"{save_name}_rec.png")

        if img_gt is not None:
            render_psnr = batch_psnr(img_org, img_raw)
            render_ssim = batch_ssim(img_org, img_raw)
            print(f"Rendered image: PSNR={render_psnr}, SSIM={render_ssim}")

            rec_psnr = batch_psnr(img_gt, img_rec)
            rec_ssim = batch_ssim(img_gt, img_rec)
            print(f"Rec image: PSNR={rec_psnr}, SSIM={rec_ssim}")

    # ====================================================================================
    # PSF (incoherent ray tracing)
    # ====================================================================================
    @torch.no_grad()
    def psf_center(self, point, method="chief_ray"):
        """Compute reference PSF center (flipped to match the original point, green light) for given point source.

        Args:
            point: [..., 3] un-normalized point is in object plane.

        Returns:
            psf_center: [..., 2] un-normalized psf center in sensor plane.
        """
        if method == "chief_ray":
            # Shrink the pupil and calculate centroid ray as the chief ray.
            ray = self.sample_from_points(point, num_rays=SPP_CALC, scale_pupil=0.2)
            ray = self.trace2sensor(ray)
            assert (ray.valid == 1).any(), "No sampled rays is valid."
            valid = ray.valid.unsqueeze(-1)
            psf_center = (ray.o * valid).sum(-2) / valid.sum(-2).add(
                EPSILON
            )  # shape [..., 3]
            psf_center = -psf_center[..., :2]  # shape [..., 2]

        elif method == "pinhole":
            # Pinhole camera perspective projection
            # Calculate the FoV of incident ray, then map to sensor plane
            tan_point_fov_x = -point[..., 0] / point[..., 2]
            tan_point_fov_y = -point[..., 1] / point[..., 2]
            tan_hfov_x = float(
                np.tan(self.hfov) * self.sensor_size[1] / self.r_sensor / 2
            )
            tan_hfov_y = float(
                np.tan(self.hfov) * self.sensor_size[0] / self.r_sensor / 2
            )
            psf_center_x = tan_point_fov_x / tan_hfov_x * self.sensor_size[1] / 2
            psf_center_y = tan_point_fov_y / tan_hfov_y * self.sensor_size[0] / 2
            psf_center = torch.stack([psf_center_x, psf_center_y], dim=-1)

        else:
            raise ValueError(f"Unsupported method: {method}.")

        return psf_center

    def psf(self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_PSF, recenter=False):
        """Single wvln incoherent PSF calculation.

        Args:
            points (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 51.
            spp (int, optional): Sample per pixel. For different ray tracing, usually kernel_size^2. Defaults to 2048.
            recenter (bool, optional): Recenter the PSF using chief ray.

        Returns:
            kernel: Shape of [ks, ks] or [N, ks, ks].
        """
        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points)
        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # Ray position in the object space by perspective projection, because points are normalized
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * self.sensor_size[1] / 2
        point_obj_y = points[..., 1] * scale * self.sensor_size[0] / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)

        # Trace rays to sensor plane
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)
        ray = self.trace2sensor(ray)

        # Calculate PSF
        if recenter:
            # PSF center on the sensor plane defined by chief ray
            pointc_chief_ray = self.psf_center(point_obj)  # shape [N, 2]
            psf = forward_integral(
                ray,
                ps=self.pixel_size,
                ks=ks,
                pointc=pointc_chief_ray,
                coherent=False,
            )
        else:
            # PSF center on the sensor plane defined by perspective projection
            pointc_ideal = points.clone()[:, :2]
            pointc_ideal[:, 0] *= self.sensor_size[1] / 2
            pointc_ideal[:, 1] *= self.sensor_size[0] / 2
            psf = forward_integral(
                ray, ps=self.pixel_size, ks=ks, pointc=pointc_ideal.to(self.device), coherent=False
            )

        # Normalize to 1
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return psf

    def psf_map(
        self,
        depth=DEPTH,
        grid=(7, 7),
        ks=PSF_KS,
        spp=SPP_PSF,
        wvln=DEFAULT_WAVE,
        recenter=True,
    ):
        """Computes the PSF map at a depth. Overrides the base method to improve efficiency by parallel ray tracing over different points.

        Args:
            depth (float, optional): Depth of the point source plane. Defaults to DEPTH.
            grid (int, optional): Grid size. Defaults to 7.
            ks (int, optional): Kernel size. Defaults to 51.
            spp (int, optional): Sample per pixel. Defaults to None.
            recenter (bool, optional): Recenter the PSF using chief ray. Defaults to True.

        Returns:
            psf_map: Shape of [grid*ks, grid*ks].
        """
        if isinstance(grid, int):
            grid = (grid, grid)
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # shape [grid**2, 1, ks, ks]

        psf_map = psfs.reshape(grid[0], grid[1], 1, ks, ks)
        return psf_map

    # ====================================================================================
    # PSF (coherent ray tracing)
    # ====================================================================================
    def pupil_field(self, point, wvln=DEFAULT_WAVE, spp=SPP_COHERENT):
        """Compute complex wavefront (flipped for further PSF calculation) at exit pupil plane by coherent ray tracing. The wavefront has the same size as image sensor.

            This function is differentiable.

        Args:
            point (tensor): Point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            wvln (float): Ray wavelength in [um].
            spp (int): Ray sample number per point.
        """
        assert spp >= 1000000, (
            "Coherent ray tracing spp is too small, will cause inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Please set the default dtype to float64 for accurate phase calculation."
        )

        if isinstance(point, list):
            point = torch.tensor(point, device=self.device).unsqueeze(
                0
            )  # shape of [1, 3]
        elif torch.is_tensor(point) and len(point.shape) == 1:
            point = point.unsqueeze(0).to(torch.float64)  # shape of [1, 3]
        else:
            raise Exception("Unsupported point type.")

        # Ray origin in the object space
        scale = self.calc_scale(point[:, 2].item())
        point_obj = point.clone()
        point_obj[:, 0] = point[:, 0] * scale * self.sensor_size[1] / 2  # x coordinate
        point_obj[:, 1] = point[:, 1] * scale * self.sensor_size[0] / 2  # y coordinate

        # Ray center determined by chief ray
        # shape [N, 2], un-normalized physical coordinates
        pointc_chief_ray = self.psf_center(point_obj)

        # Ray-tracing to last surface
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)
        ray.coherent = True
        ray, _ = self.trace(ray)

        # Back-trace to exit pupil plane
        pupilz, pupilr = self.calc_exit_pupil()
        ray = ray.prop_to(pupilz)

        # Calculate a full-resolution complex field for exit-pupil diffraction
        pointc_ref = torch.zeros_like(point[:, :2]).to(self.device)
        wavefront = forward_integral(
            ray,
            ps=self.pixel_size,
            ks=self.sensor_res[0],
            pointc=pointc_ref,
            coherent=True,
        )
        wavefront = wavefront.squeeze(0)  # shape of [H, W]

        # Aperture clip

        # PSF center (on the sensor plane)
        pointc_chief_ray = pointc_chief_ray[0, :]
        psf_center = [
            pointc_chief_ray[0] / self.sensor_size[0] * 2,
            pointc_chief_ray[1] / self.sensor_size[1] * 2,
        ]

        return wavefront, psf_center

    def psf_coherent(self, point, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT):
        """Single point monochromatic PSF using ray-wave model.

        Steps:
            1, calculate complex wavefield at DOE (pupil) plane by coherent ray tracing.
            2, propagate through DOE to sensor plane, calculate intensity PSF, crop the valid region and normalize.

        Args:
            point (torch.Tensor, optional): [x, y, z] coordinates of the point source. Defaults to torch.Tensor([0,0,-10000]).
            ks (int, optional): size of the PSF patch. Defaults to 101.
            wvln (float, optional): wvln. Defaults to 0.589.
            spp (int, optional): number of rays to sample. Defaults to 1000000.

        Returns:
            psf_out (torch.Tensor): PSF patch. Normalized to sum to 1. Shape [ks, ks]
        """
        # Pupil field by coherent ray tracing
        wavefront, psfc = self.pupil_field(point=point, wvln=wvln, spp=spp)

        # Propagate to sensor and get intensity. (Manually pad wave field)
        pupilz, pupilr = self.calc_exit_pupil()
        h, w = wavefront.shape
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=self.d_sensor - pupilz,
            wvln=wvln,
            ps=self.pixel_size,
            padding=False,
        )

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

        psf /= psf.sum()  # shape of [ks, ks] or [h, w]
        psf = diff_float(psf)
        return psf

    # ====================================================================================
    # Classical optical design
    # ====================================================================================
    def analysis_rms(self, depth=float("inf")):
        """Compute RMS spot size and radius for on-axis and off-axis fields.

        Args:
            depth (float, optional): Depth of the point source. Defaults to float("inf").
        """
        num_field = 3
        rms_error_fields = []
        rms_radius_fields = []
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
            ray_xy_norm = (ray.o[..., :2] - ray_xy_center_green) * ray.valid.unsqueeze(-1)
            rms_error = ((ray_xy_norm**2).sum(-1).sqrt() * ray.valid).sum(-1) / (
                ray.valid.sum(-1) + EPSILON
            )
            rms_radius = (ray_xy_norm**2).sum(-1).sqrt().max(dim=-1).values

            # Append to list
            rms_error_fields.append(rms_error)
            rms_radius_fields.append(rms_radius)

        # Average over wavelengths
        avg_rms_error_um = torch.stack(rms_error_fields).mean(dim=0) * 1000.0
        avg_rms_radius_um = torch.stack(rms_radius_fields).mean(dim=0) * 1000.0

        print(
            f"RMS average error (chief ray): center {avg_rms_error_um[0]:.3f} um, middle {avg_rms_error_um[1]:.3f} um, off-axis {avg_rms_error_um[-1]:.3f} um"
        )
        print(
            f"RMS maximum radius (chief ray): center {avg_rms_radius_um[0]:.3f} um, middle {avg_rms_radius_um[1]:.3f} um, off-axis {avg_rms_radius_um[-1]:.3f} um"
        )

    # ====================================================================================
    # Geometrical optics calculation
    # ====================================================================================
    def find_aperture(self):
        """Find aperture. If the lens has no aperture, use the surface with the smallest radius."""
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Aperture):
                self.aper_idx = i
                return

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])

    def find_diff_surf(self):
        """Get differentiable/optimizable surface list."""
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )
        return diff_surf_range

    @torch.no_grad()
    def calc_foclen(self):
        """Calculate the effective focal length."""
        return self.calc_efl()

    @torch.no_grad()
    def calc_efl(self):
        """Compute effective focal length (EFL).

        Trace a paraxial chief ray and compute the image height, then use the image height to compute the EFL.

        Reference:
            [1] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Tutorial_MorelSophie.pdf
        """
        # Trace a paraxial chief ray, shape [1, 1, num_rays, 3]
        paraxial_fov_rad = 0.001
        paraxial_fov_deg = float(np.rad2deg(paraxial_fov_rad))
        self.calc_pupil()
        ray = self.sample_parallel(fov_x=0.0, fov_y=paraxial_fov_deg, scale_pupil=0.2)
        ray = self.trace2sensor(ray)

        # Compute the effective focal length
        paraxial_imgh = (ray.o[0, 0, :, 1] * ray.valid[0, 0, :]).sum() / ray.valid[0, 0, :].sum()
        eff_foclen = paraxial_imgh.item() / float(np.tan(paraxial_fov_rad))
        return eff_foclen

    @torch.no_grad()
    def calc_bfl(self, wvln=DEFAULT_WAVE):
        """Compute back focal length (BFL).

        BFL: Distance from the second principal point to focal plane.

        FIXME: this definition is not correct.
        """
        # Forward ray tracing
        ray = self.sample_parallel(fov_x=0.0, fov_y=0.0, num_rays=SPP_CALC, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _ = self.trace(ray)

        # Principal point
        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z_principal = out_ray.o[..., 2] - out_ray.d[..., 2] * t

        # Focal point
        t = -out_ray.o[..., 0] / out_ray.d[..., 0]
        z_focus = out_ray.o[..., 2] + out_ray.d[..., 2] * t

        # Back focal length
        bfl = z_focus - z_principal
        bfl = float(np.nanmean(bfl[ray.valid > 0].cpu().numpy()))

        return bfl

    @torch.no_grad()
    def calc_eqfl(self):
        """35mm equivalent focal length. For cellphone lens, we usually use EFL to describe the lens.

        35mm sensor: 36mm * 24mm
        """
        return 21.63 / math.tan(self.hfov)

    @torch.no_grad()
    def calc_fnum(self):
        """Compute f-number."""
        _, pupilr = self.get_entrance_pupil()
        return self.calc_efl() / (2 * pupilr)

    @torch.no_grad()
    def calc_numerical_aperture(self):
        """Compute numerical aperture."""
        raise NotImplementedError("Numerical aperture is not implemented.")

    @torch.no_grad()
    def calc_foc_dist(self, wvln=DEFAULT_WAVE):
        """Compute the focus depth in the object space of the lens. Ray starts from sensor and trace to the object space.

        Returns:
            focus_dist: Focus distance in object space. Negative value.
        """
        # Sample point source rays from sensor center
        o1 = torch.tensor([0, 0, self.d_sensor.item()], device=self.device).repeat(
            SPP_CALC, 1
        )

        # Sample the first surface as pupil
        o2 = self.sample_circle(self.surfaces[0].r, z=0.0, shape=[SPP_CALC]).to(
            self.device
        )
        o2 *= 0.25  # Shrink sample region to improve accuracy
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=self.device)

        # Trace rays to object space
        ray, _ = self.trace(ray)

        # Optical axis intersection
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_p = (ray.o[..., 2] - ray.d[..., 2] * t)[ray.valid > 0].cpu().numpy()
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p < 0)]

        if len(focus_p) > 0:
            focus_dist = float(np.mean(focus_p))
        else:
            raise Exception("No valid focus distance is found.")

        return focus_dist

    @torch.no_grad()
    def calc_foc_plane(self, depth=float("inf")):
        """Calculate in-focus sensor plane."""
        # Sample and trace rays
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
        ray, _ = self.trace(ray)

        # Calculate in-focus sensor position
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_p = ray.o[..., 2] - ray.d[..., 2] * t
        focus_p = focus_p[ray.valid > 0]
        focus_p = focus_p[~torch.isnan(focus_p) & (focus_p > 0)]
        infocus_sensor_d = torch.mean(focus_p)

        return infocus_sensor_d

    @torch.no_grad()
    def calc_hfov(self):
        """Compute half diagonal fov.

        Shot rays from edge of sensor, trace them to the object space and compute output angel as the fov.
        """
        # Sample rays going out from edge of sensor, shape [SPP_CALC, 3]
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

        # Compute fov from output ray direction
        tan_hfov = ray.d[..., 0] / ray.d[..., 2]
        hfov = torch.atan(torch.sum(tan_hfov * ray.valid) / torch.sum(ray.valid))

        # If calculation failed, use pinhole camera model to compute fov
        if torch.isnan(hfov):
            hfov = float(np.arctan(self.r_sensor / self.foclen))
            print(f"Computing fov failed, set fov to {hfov}.")
        else:
            hfov = hfov.item()

        # Compute horizontal (x) and vertical (y) fov from diagonal fov
        diag = math.sqrt(self.sensor_size[0] ** 2 + self.sensor_size[1] ** 2)
        self.hfov_x = math.atan((self.sensor_size[0] * math.tan(hfov)) / diag)
        self.hfov_y = math.atan((self.sensor_size[1] * math.tan(hfov)) / diag)
        self.hfov = hfov

        return hfov

    @torch.no_grad()
    def calc_scale(self, depth):
        """Calculate the scale factor (obj_height / img_height) with pinhole camera model."""
        return -depth / self.foclen

    # @torch.no_grad()
    # def calc_scale_pinhole(self, depth):
    #     """Scale factor computed by pinhole camera model.

    #     This function assumes the first principal point is at (0, 0, 0) and the second principal point is at (0, 0, d_sensor - focal_length).

    #     Args:
    #         depth (float): depth of the object.

    #     Returns:
    #         scale (float): scale factor. phy_size_obj / phy_size_img
    #     """
    #     return -depth / self.foclen

    # @torch.no_grad()
    # def calc_scale_ray(self, depth):
    #     """Use ray tracing to compute scale factor."""
    #     if isinstance(depth, float) or isinstance(depth, int):
    #         # Sample rays [num_grid, num_grid, spp, 3] from the object plane
    #         num_grid = 64
    #         raise Warning(
    #             "This function needs to be checked because of the change of the ray sampling function."
    #         )
    #         ray = self.sample_point_source(
    #             depth=depth, num_rays=SPP_CALC, num_grid=num_grid
    #         )

    #         # Map points from object space to sensor space, ground-truth
    #         o1 = ray.o.clone()[..., :2]
    #         o1 = torch.flip(o1, [0, 1])

    #         ray, _ = self.trace(ray)
    #         o2 = ray.project_to(self.d_sensor)  # shape [num_grid, num_grid, spp, 2]

    #         # Use only center region of points, because we assume center points have no distortion
    #         center_start = num_grid // 2 - num_grid // 8
    #         center_end = num_grid // 2 + num_grid // 8
    #         o1_center = o1[center_start:center_end, center_start:center_end, :, :]
    #         o2_center = o2[center_start:center_end, center_start:center_end, :, :]
    #         ra_center = ray.valid.clone().detach()[
    #             center_start:center_end, center_start:center_end, :
    #         ]

    #         x1 = o1_center[:, :, 0, 0]  # shape [num_grid // 4, num_grid // 4]
    #         x2 = (o2_center[:, :, :, 0] * ra_center).sum(dim=-1) / (ra_center).sum(
    #             dim=-1
    #         ).add(EPSILON)

    #         # Calculate scale factor (currently assume rotationally symmetric)
    #         scale_x = x1 / x2  # shape [num_grid // 4, num_grid // 4]
    #         try:
    #             scale = torch.mean(scale_x[~scale_x.isnan()]).item()
    #         except Exception as e:
    #             print(f"Error calculating scale: {e}")
    #             scale = -depth * np.tan(self.hfov) / self.r_sensor
    #         return scale

    #     elif isinstance(depth, torch.Tensor) and len(depth.shape) == 1:
    #         scale = []
    #         for d in depth:
    #             scale.append(self.calc_scale_ray(d.item()))
    #         scale = torch.tensor(scale)
    #         return scale

    #     else:
    #         raise ValueError("Invalid depth type.")

    @torch.no_grad()
    def chief_ray(self):
        """Compute chief ray from sensor to object space.

        We can use chief ray for fov, magnification. Chief ray, a ray goes through center of aperture.

        This function is currently not used and needs to be checked.
        """
        # sample rays with shape [SPP_CALC, 3]
        pupilz, pupilx = self.get_exit_pupil()
        o1 = torch.zeros([SPP_CALC, 3])
        o1[:, 0] = pupilx
        o1[:, 2] = self.d_sensor.item()

        x2 = torch.linspace(-pupilx, pupilx, SPP_CALC)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        inc_ray = ray.clone()
        ray, _ = self.trace(
            ray, lens_range=list(range(self.aper_idx, len(self.surfaces)))
        )

        center_x = torch.min(torch.abs(ray.o[:, 0]))
        center_idx = torch.where(torch.abs(ray.o[:, 0]) == center_x)

        return inc_ray.o[center_idx, :], inc_ray.d[center_idx, :]

    @torch.no_grad()
    def calc_pupil(self, paraxial=False):
        """
        Compute entrance and exit pupil positions and radii.
        The entrance and exit pupils must be recalculated whenever:
            - First-order parameters change (e.g., field of view, object height, image height),
            - Lens geometry or materials change (e.g., surface curvatures, refractive indices, thicknesses),
            - Or generally, any time the lens configuration is modified.

        Args:
            paraxial (bool): If True, use paraxial approximation. Default: True.

        Notes:
            - If `self.float_enpd` is True, set ENPD based on computed pupil radius.
            - Otherwise, override computed entrance pupil radius using fixed ENPD.
        """
        self.find_aperture()
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil(paraxial)
        self.entrance_pupilz, self.entrance_pupilr = self.calc_entrance_pupil(paraxial)

    def get_entrance_pupil(self, scale_pupil=1.0):
        """
        Get entrance pupil location and radius with optional scaling.

        Args:
            scale_pupil (float): Scale factor for pupil radius. Default: 1.

        Returns:
            tuple: (z position, radius) of entrance pupil.
        """
        entrance_pupilz, entrance_pupilr = (
            self.entrance_pupilz,
            self.entrance_pupilr * scale_pupil,
        )
        return entrance_pupilz, entrance_pupilr

    def get_exit_pupil(self, scale_pupil=1.0):
        """
        Get exit pupil location and radius with optional scaling.
        The exit pupils must be recalculated when the lens is modified.

        Args:
            scale_pupil (float): Scale factor for pupil radius. Default: 1.

        Returns:
            tuple: (z position, radius) of exit pupil.
        """
        exit_pupilz, exit_pupilr = self.exit_pupilz, self.exit_pupilr * scale_pupil
        return exit_pupilz, exit_pupilr

    @torch.no_grad()
    def calc_exit_pupil(self, paraxial=False):
        """
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
            phi = torch.linspace(-0.1, 0.1, 32) / 180.0 * torch.pi
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)
            phi = torch.linspace(-0.5, 0.5, SPP_CALC)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to last surface
        lens_range = range(self.aper_idx + 1, len(self.surfaces))
        ray, _ = self.trace(ray, lens_range=lens_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.valid != 0][:, 0], ray.o[ray.valid != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.valid != 0][:, 0], ray.d[ray.valid != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small pupil
        if len(intersection_points) == 0:
            print("No intersection points found, use the last surface as pupil.")
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
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            print("No aperture, use the first surface as entrance pupil.")
            return self.surfaces[0].d.item(), self.surfaces[0].r

        # Sample rays from aperture stop
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]]).repeat(32, 1)
            phi = torch.linspace(-0.1, 0.1, 32) / 180.0 * torch.pi
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)
            phi = torch.linspace(-0.25, 0.25, SPP_CALC)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to first surface
        lens_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, lens_range=lens_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.valid > 0][:, 0], ray.o[ray.valid > 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.valid > 0][:, 0], ray.d[ray.valid > 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small entrance pupil
        if len(intersection_points) == 0:
            print("Calculate entrance pupil failed, use the first surface.")
            avg_pupilr = self.surfaces[0].r
            avg_pupilz = self.surfaces[0].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface."
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
        x, _ = torch.linalg.lstsq(
            A,
            b.unsqueeze(-1),
        )[:2]
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
        d_sensor_new = self.calc_foc_plane(depth=foc_dist)

        # Update sensor position
        assert d_sensor_new > 0, "sensor position is negative."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.update_float_setting()

    @torch.no_grad()
    def set_aperture(self, fnum=None, foclen=None, aper_r=None):
        """Change aperture radius."""
        raise Exception("This function will be deprecated in the future.")
        if aper_r is None:
            if foclen is None:
                foclen = self.calc_efl()
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r

        self.fnum = self.foclen / aper_r / 2

    @torch.no_grad()
    def set_fnum(self, fnum):
        """Set F number and aperture radius using binary search.

        Args:
            fnum (float): F number.
        """
        target_pupil_r = self.foclen / fnum / 2

        # Binary search to find aperture radius that gives desired exit pupil radius
        optim_aper_r = target_pupil_r
        aper_r_min = 0.5 * target_pupil_r
        aper_r_max = 2.0 * target_pupil_r

        for _ in range(8):
            self.surfaces[self.aper_idx].r = optim_aper_r
            _, pupilr = self.get_entrance_pupil()

            if abs(pupilr - target_pupil_r) < 0.1:  # Close enough
                break

            if pupilr > target_pupil_r:
                # Current radius is too large, decrease it
                aper_r_max = optim_aper_r
                optim_aper_r = (aper_r_min + optim_aper_r) / 2
            else:
                # Current radius is too small, increase it
                aper_r_min = optim_aper_r
                optim_aper_r = (aper_r_max + optim_aper_r) / 2

        self.surfaces[self.aper_idx].r = optim_aper_r

    @torch.no_grad()
    def set_target_fov_fnum(self, hfov, fnum, foclen):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        Args:
            hfov (float): half diagonal-FoV in radian.
            fnum (float): F number.
            foclen (float): focal length in [mm].
        """
        self.hfov = hfov
        self.fnum = fnum
        # ======>
        # self.foclen = foclen
        # ======>
        self.foclen = self.r_sensor / math.tan(hfov)
        # ======>

        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].update_r(float(aper_r))

    @torch.no_grad()
    def set_fov(self, hfov):
        """Set FoV. This function is used to assign design targets.

        Args:
            hfov (float): half diagonal-FoV in degree.
        """
        self.hfov = hfov

    @torch.no_grad()
    def set_sensor(self, sensor_res, sensor_size=None, r_sensor=None):
        """Set four parameters of camera sensor: resolution, size, r_sensor and pixel size.

        Args:
            sensor_res: Resolution, pixel number.
            sensor_size: Sensor size in [mm].
            r_sensor: Sensor radius in [mm].
        """
        if sensor_size is not None:
            assert r_sensor is None, (
                "Sensor_size is provided, no need to provide r_sensor."
            )
            assert sensor_res[0] * sensor_size[1] == sensor_res[1] * sensor_size[0], (
                "sensor_res and sensor_size are not consistent"
            )

            self.sensor_res = sensor_res
            self.sensor_size = sensor_size
            self.r_sensor = math.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2) / 2
            self.pixel_size = sensor_size[0] / sensor_res[0]

            self.update_float_setting()

        elif r_sensor is not None:
            assert sensor_size is None, (
                "sensor_res and r_sensor are provided, no need to provide sensor_size."
            )
            if isinstance(sensor_res, int):
                sensor_res = (sensor_res, sensor_res)
            self.sensor_res = sensor_res
            self.r_sensor = r_sensor
            self.sensor_size = [
                2
                * self.r_sensor
                * sensor_res[0]
                / math.sqrt(sensor_res[0] ** 2 + sensor_res[1] ** 2),
                2
                * self.r_sensor
                * sensor_res[1]
                / math.sqrt(sensor_res[0] ** 2 + sensor_res[1] ** 2),
            ]
            self.pixel_size = self.sensor_size[0] / self.sensor_res[0]

            self.update_float_setting()

        else:
            raise ValueError(
                "Either sensor_res or r_sensor must be provided, and both cannot be provided at the same time."
            )

    @torch.no_grad()
    def prune_surf(self, expand_factor=None):
        """Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            expand_factor (float): extra height to reserve.
                - For cellphone lens, we usually use 0.1mm or 0.05 * r_sensor.
                - For camera lens, we usually use 0.5mm or 0.1 * r_sensor.
            surface_range (list): surface range to prune.
        """
        # Settings
        surface_range = self.find_diff_surf()

        if self.r_sensor < 10.0:
            expand_factor = 0.05 if expand_factor is None else expand_factor

            # # Reset lens to maximum height(sensor radius)
            # for i in surface_range:
            #     # self.surfaces[i].r = self.r_sensor
            #     self.surfaces[i].r = max(self.r_sensor, self.surfaces[self.aper_idx].r)
        else:
            expand_factor = 0.4 if expand_factor is None else expand_factor

        # Sample maximum fov rays to cut valid surface height
        if self.hfov is not None:
            fov_deg = self.hfov * 180 / torch.pi
        else:
            fov_deg = (
                float(np.arctan(self.r_sensor / self.d_sensor.item())) * 180 / torch.pi
            )

        # Trace rays to compute the maximum valid region of the lens, shape of ray: [num_rays, 3]
        ray = self.sample_parallel(fov_x=[0.0], fov_y=[fov_deg], num_rays=SPP_CALC)
        ray = ray.squeeze(0).squeeze(0)
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(ray_o_record, 0.0)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]
        for i in surface_range:
            # Determine and update surface height
            if surf_r_max[i] > 0:
                max_height_expand = surf_r_max[i].item() * (1 + expand_factor)
                max_height_allowed = self.surfaces[i].max_height()
                self.surfaces[i].update_r(min(max_height_expand, max_height_allowed))
            else:
                print(f"No valid rays for Surf {i}, do not prune.")
                max_height_expand = self.surfaces[i].r * (1 + expand_factor)
                max_height_value_range = self.surfaces[i].max_height()
                self.surfaces[i].update_r(min(max_height_expand, max_height_value_range))

    @torch.no_grad()
    def correct_shape(self, expand_factor=None):
        """Correct wrong lens shape during the lens design."""
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
            d_aper = 0.05 if self.is_cellphone else 1.0

            # If the first surface is concave, use the maximum negative sag.
            aper_r = torch.tensor(self.surfaces[aper_idx].r, device=self.device)
            sag1 = -self.surfaces[aper_idx + 1].sag(aper_r, 0).item()

            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in optim_surf_range:
                self.surfaces[i].d -= delta_aper

        # Rule 4: Prune all surfaces
        self.prune_surf(expand_factor=expand_factor)

        if shape_changed:
            print("Surface shape corrected.")
        return shape_changed

    # ====================================================================================
    # Manufacturing and tolerance analysis
    # ====================================================================================
    @torch.no_grad()
    def perturb(self):
        """Randomly perturb all lens surfaces to simulate manufacturing errors.

        Including:
            (1) surface position, thickness, curvature, and other coefficients.
            (2) surface rotation, tilt, and decenter.

        Called for accurate image simulation, together with sensor noise, vignetting, etc.
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].perturb()

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match material"""
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)

    @torch.no_grad()
    def analysis_tolerance(self):
        """Analyze tolerance"""
        pass

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
    ):
        """Analyze the optical lens.

        Args:
            save_name (str): save name.
            depth (float): object depth distance.
            render (bool): whether render an image.
            render_unwarp (bool): whether unwarp the rendered image.
            lens_title (str): lens title
        """
        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
        )

        # Draw spot diagram
        self.draw_spot_radial(depth=depth, save_name=f"{save_name}_spot.png")

        # Draw MTF
        if depth == float("inf"):
            # This is a hack to draw MTF for infinite depth
            self.draw_mtf(depth_list=[DEPTH], save_name=f"{save_name}_mtf.png")
        else:
            self.draw_mtf(depth_list=[depth], save_name=f"{save_name}_mtf.png")

        # Calculate RMS error
        self.analysis_rms(depth=depth)

        # Render an image, compute PSNR and SSIM
        if render:
            depth = DEPTH if depth == float("inf") else depth
            img_org = cv.cvtColor(cv.imread("./datasets/IQ/img1.png"), cv.COLOR_BGR2RGB)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                noise=0.01,
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
            optim_surf_range = self.find_diff_surf()
        
        # If lr for each surface is a list is given
        if isinstance(lrs[0], list):
            return self.get_optimizer_params_manual(lrs=lrs, optim_mat=optim_mat, optim_surf_range=optim_surf_range)

        # Optimize lens surface parameters
        params = []
        for surf_idx in optim_surf_range:
            surf = self.surfaces[surf_idx]

            if isinstance(surf, Aperture):
                params += surf.get_optimizer_params(lrs=[lrs[0]])

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(lrs=lrs[:4], decay=decay, optim_mat=optim_mat)

            elif isinstance(surf, AsphericNorm):
                params += surf.get_optimizer_params(lrs=lrs[:4], decay=decay, optim_mat=optim_mat)

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
                params += surf.get_optimizer_params(lrs=[lrs[0], lrs[1]], optim_mat=optim_mat)

            elif isinstance(surf, ThinLens):
                params += surf.get_optimizer_params(lrs=[lrs[0], lrs[1]], optim_mat=optim_mat)

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
        params = self.get_optimizer_params(lrs=lrs, decay=decay, optim_surf_range=optim_surf_range, optim_mat=optim_mat)
        optimizer = torch.optim.Adam(params)
        # optimizer = torch.optim.SGD(params)
        return optimizer

    # ====================================================================================
    # Lens file IO
    # ====================================================================================
    def read_lens_json(self, filename="./test.json"):
        """Read the lens from .json file."""
        self.surfaces = []
        self.materials = []
        with open(filename, "r") as f:
            data = json.load(f)
            d = 0.0
            for surf_dict in data["surfaces"]:
                surf_dict["d"] = d

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
        self.float_hfov = False

        sensor_res = data.get("sensor_res", self.sensor_res)
        self.r_sensor = data["r_sensor"]

        self.to(self.device)
        self.set_sensor(sensor_res=sensor_res, r_sensor=self.r_sensor)

    def write_lens_json(self, filename="./test.json"):
        """Write the lens into .json file."""
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
            surf_dict = {"idx": i + 1}
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

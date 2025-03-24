"""A geometric lens consisting of refractive surfaces, simulate with ray tracing. May contain diffractive surfaces, but still use ray tracing to simulate.

For image simulation:
    1. Ray tracing based rendering
    2. PSF + patch convolution

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import json
import logging
import math
import os
import random
from datetime import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import LstsqConvert
LstsqConvert.register_op()

from .lens import Lens
from .optics.basics import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
    init_device,
)
from .optics.materials import SELLMEIER_TABLE, Material
from .optics.monte_carlo import forward_integral
from .optics.ray import Ray
from .optics.surfaces import (
    Anamorphic,
    Aperture,
    Aspheric,
    Cubic,
    Diffractive_GEO,
    Plane,
    Spheric,
    ThinLens,
)
from .optics.wave import AngularSpectrumMethod
from .optics.waveoptics_utils import diff_float
from .utils import (
    batch_psnr,
    batch_ssim,
    denormalize_ImageNet,
    img2batch,
    normalize_ImageNet,
    set_logger,
)


class GeoLens(Lens):
    """Geolens class. A geometric lens consisting of refractive surfaces, simulate with ray tracing. May contain diffractive surfaces, but still use ray tracing to simulate."""

    def __init__(self, filename=None):
        """Initialize a geometric lens."""
        self.device = init_device()

        # Load lens file
        if filename is not None:
            self.lens_name = filename
            self.load_file(filename)
            self.to(self.device)

            # Lens calculation
            self.find_aperture()
            self.post_computation()

        else:
            self.sensor_res = [1024, 1024]
            self.surfaces = []
            self.materials = []
            self.to(self.device)

    def load_file(self, filename):
        """Load lens file."""
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

    def double(self):
        """Use double-precision for coherent ray tracing."""
        torch.set_default_dtype(torch.float64)
        for surf in self.surfaces:
            surf.double()

    def post_computation(self):
        """After loading lens, compute foclen, fov and fnum."""
        self.find_aperture()

        self.hfov = self.calc_hfov()
        self.foclen = self.calc_efl()
        self.fnum = self.calc_fnum()

        self.init_constraints()

    # ====================================================================================
    # Ray sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(
        self,
        fov=0.0,
        depth=0.0,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        entrance_pupil=False,
        forward=True,
    ):
        """Sample 2D parallel rays. Used for (1) drawing lens setup, (2) 2D geometric optics calculation, for example, refocusing to infinity

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to 0.0.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            num_rays (int, optional): ray number. Defaults to 15.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilx = self.calc_entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        x = torch.linspace(-pupilx, pupilx, num_rays) * 0.99
        y = torch.zeros_like(x)
        z = torch.full_like(x, pupilz)
        ray_o = torch.stack((x, y, z), axis=-1)  # shape [num_rays, 3]

        # Sample ray directions
        if forward:
            dx = torch.full_like(x, np.sin(fov / 57.3))
            dy = torch.zeros_like(x)
            dz = torch.full_like(x, np.cos(fov / 57.3))
        else:
            dx = torch.full_like(x, -np.sin(fov / 57.3))
            dy = torch.zeros_like(x)
            dz = torch.full_like(x, -np.cos(fov / 57.3))

        ray_d = torch.stack((dx, dy, dz), axis=-1)  # shape [num_rays, 3]

        # Form rays
        rays = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        rays.propagate_to(depth)
        return rays

    @torch.no_grad()
    def sample_parallel(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=0.0,
        num_rays=SPP_CALC,
        wvln=DEFAULT_WAVE,
        entrance_pupil=False,
    ):
        """Sample parallel rays from object space. Returns rays for each combination of fov_x and fov_y. Used for geometric optics calculation.

        Args:
            fov_x (float or list): angle rotated from z-axis to ray direction in x0z plane, positive is clockwise.
            fov_y (float or list): angle rotated from z-axis to ray direction in y0z plane, positive is clockwise.
            depth (float, optional): sampling depth. Defaults to 0.0.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [num_fov_x, num_fov_y, num_rays, 3]
        """
        if isinstance(fov_x, float):
            fov_x = [fov_x]
        if isinstance(fov_y, float):
            fov_y = [fov_y]

        # Create meshgrid of fov angles
        fx_grid, fy_grid = torch.meshgrid(
            torch.tensor([fx / 57.3 for fx in fov_x]),
            torch.tensor([fy / 57.3 for fy in fov_y]),
            indexing="ij",
        )

        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        ray_o = self.sample_circle(
            pupilr, pupilz, shape=[len(fov_x), len(fov_y), num_rays]
        )  # [num_fov_x, num_fov_y, num_rays, 3]

        # Calculate ray directions
        dx = torch.tan(-fx_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
        dy = torch.tan(-fy_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
        dz = torch.ones_like(ray_o[..., 2])
        ray_d = torch.stack((dx, dy, dz), dim=-1)  # [num_fov_x, num_fov_y, num_rays, 3]

        # Form rays
        rays = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        rays.propagate_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self,
        fov=0.0,
        depth=DEPTH,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        entrance_pupil=False,
    ):
        """Sample point source 2D rays. Used for (1) drawing lens setup.

        Args:
            depth (float, optional): sampling depth.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample point on the object plane
        ray_o = torch.tensor(
            [depth * float(np.tan(fov / 57.3)), 0, depth]
        )
        ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)

        # Sample points (second point) on the pupil
        if entrance_pupil:
            pupilz, pupilx = self.calc_entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        x2 = torch.linspace(-pupilx, pupilx, num_rays) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        ray_o2 = torch.stack((x2, y2, z2), axis=1)

        # Form the rays
        ray_d = ray_o2 - ray_o
        ray = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        ray.propagate_to(depth)
        return ray

    @torch.no_grad()
    def sample_point_source(
        self,
        depth=DEPTH,
        num_grid=[11, 11],
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        importance_sampling=False,
    ):
        """Sample forward rays from 2D grid in the object space. Used for (1) spot/rms/magnification calculation, (2) distortion/sensor sampling

        This function is equivalent to self.point_source_grid() + self.sample_from_points().

        Args:
            depth (float, optional): sample plane z position. Defaults to -10.0.
            num_grid (int or list, optional): sample plane resolution. Defaults to 11.
            num_rays (int, optional): number of rays sampled from each grid point. Defaults to 16.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid, num_grid, num_rays, 3]
        """
        if isinstance(num_grid, int):
            num_grid = [num_grid, num_grid]

        # Sample normalized grid points [-1, 1] * [-1, 1] on the sensor plane
        x, y = torch.meshgrid(
            torch.linspace(
                -1 / 2,
                1 / 2,
                num_grid[1],
                device=self.device,
            ),
            torch.linspace(
                -1 / 2,
                1 / 2,
                num_grid[0],
                device=self.device,
            ),
            indexing="xy",
        )

        # Importance sampling to concentrate rays towards edge
        if importance_sampling:
            x = torch.sqrt(x.abs()) * x.sign()
            y = torch.sqrt(y.abs()) * y.sign()

        # Scale grid points to the object space
        scale = self.calc_scale_pinhole(depth=depth)
        x, y = x * self.sensor_size[1] * scale, y * self.sensor_size[0] * scale

        # Form ray origins
        z = torch.full_like(x, depth)
        ray_o = torch.stack((x, y, z), -1)
        ray_o = ray_o.unsqueeze(2).repeat(
            1, 1, num_rays, 1
        )  # shape [num_grid, num_grid, num_rays, 3]

        # Sample second points on the pupil
        pupilz, pupilr = self.calc_entrance_pupil()
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(num_grid[0], num_grid[1], num_rays)
        ).to(self.device)  # shape [num_grid, num_grid, num_rays, 3]

        # Compute ray directions
        ray_d = ray_o2 - ray_o

        ray = Ray(ray_o, ray_d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        shrink_pupil=False,
        normalized=False,
    ):
        """Sample forward rays from given point source (un-normalized positions). Used for (1) PSF calculation, (2) chief ray calculation.

        Args:
            points (list): ray origin. Shape [3], [N, 3], [Nx, Ny, 3]
            num_rays (int): sample per pixel. Defaults to 8.
            forward (bool): forward or backward rays. Defaults to True.
            pupil (bool): whether to use pupil. Defaults to True.
            fov (float): cone angle. Defaults to 10.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray: Ray object. Shape [*shape_points, num_rays, 3]
        """
        if normalized:
            raise NotImplementedError(
                "Currently only support unnormalized object point positions."
            )
        else:
            ray_o = torch.tensor(points) if not torch.is_tensor(points) else points

        # Sample second points on the pupil
        pupilz, pupilr = self.calc_entrance_pupil(shrink_pupil=shrink_pupil)
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*ray_o.shape[:-1], num_rays)
        ).to(ray_o.device)

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
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*self.sensor_res, spp)
        ).to(self.device)

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

    @staticmethod
    def sample_circle(r, z, shape=[16, 16, 512]):
        """Sample points inside a circle.

        Args:
            r (float): Radius of the circle.
            z (float): Z-coordinate for all sampled points.
            shape (list): Shape of the output tensor.

        Returns:
            torch.Tensor: Tensor of shape [*shape, 3] containing sampled points.
        """
        # Generate random angles
        theta = torch.rand(*shape) * 2 * torch.pi

        # Generate random radii with square root for uniform distribution
        r2 = torch.rand(*shape) * r**2
        radius = torch.sqrt(r2 + EPSILON)

        # Convert to Cartesian coordinates
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)

        # Stack to form 3D points
        points = torch.stack((x, y, z_tensor), dim=-1)

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
        is_forward = ray.d.reshape(-1, 3)[0, 2] > 0
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))

        if is_forward:
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
        ray = ray.propagate_to(depth)
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
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.propagate_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            ray_o[ray.ra == 0] = float("nan")
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
            # A hack for the case of infinite object
            if ray.o[..., 2].min() < -1000.0:
                ray.propagate_to(-0.1)
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
                ray_out_o[ray.ra == 0] = float("nan")
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
                ray_out_o[ray.ra == 0] = float("nan")
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
            self.change_sensor_res(sensor_res=(H, W))

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
        scale = self.calc_scale(depth=depth, method="pinhole")
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
        ray = ray.propagate_to(depth)
        p = ray.o[..., :2]
        pixel_size = scale * self.pixel_size
        ray.ra = (
            ray.ra
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
            image = torch.sum(irr_img * ray.ra, -1) / (torch.sum(ray.ra, -1) + EPSILON)
        else:
            image = torch.sum(irr_img * ray.ra, -1) / torch.numel(ray.ra)

        return image

    def unwarp(self, img, depth=DEPTH, grid_size=128, crop=True):
        """Unwarp rendered images using distortion map.

        Args:
            img (tensor): Rendered image tensor. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            grid_size (int, optional): Grid size. Defaults to 256.
            crop (bool, optional): Whether to crop the image. Defaults to True.

        Returns:
            img_unwarpped (tensor): Unwarped image tensor. Shape of [N, C, H, W].
        """
        # Calculate distortion grid
        distortion_grid = self.distortion(
            depth=depth, grid_size=grid_size
        )  # shape (grid_size, grid_size, 2)

        # Interpolate distortion grid to image resolution
        distortion_grid = distortion_grid.permute(2, 0, 1).unsqueeze(1)
        distortion_grid = F.interpolate(
            distortion_grid, img.shape[-2:], mode="bilinear", align_corners=True
        )
        distortion_grid = distortion_grid.permute(1, 2, 3, 0).repeat(
            img.shape[0], 1, 1, 1
        )  # shape (N, H, W, 2)

        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(
            img, distortion_grid, align_corners=True
        )  # shape (N, C, H, W)
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
        self.change_sensor_res(sensor_res=img.shape[-2:])

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
        self.change_sensor_res(sensor_res=sensor_res_original)

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
            point: [N, 3] un-normalized point is in object plane.

        Returns:
            psf_center: [N, 2] un-normalized psf center in sensor plane.
        """
        if method == "chief_ray":
            # Shrink the pupil and calculate centroid ray as the chief ray. Distortion will affect the result.
            ray = self.sample_from_points(point, num_rays=SPP_CALC, shrink_pupil=True)
            ray = self.trace2sensor(ray)
            assert (ray.ra == 1).any(), "No sampled rays is valid."
            ra = ray.ra.unsqueeze(-1)
            psf_center = (ray.o * ra).sum(-2) / ra.sum(-2).add(EPSILON)  # shape [N, 3]
            psf_center = -psf_center[..., :2]  # shape [N, 2]

        elif method == "pinhole":
            # Pinhole camera perspective projection. Assume no distortion.
            scale = self.calc_scale_pinhole(point[..., 2])
            psf_center = point[..., :2] / scale.unsqueeze(-1)

        else:
            raise ValueError(f"Unsupported method: {method}.")

        return psf_center

    def psf(self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_PSF, recenter=True):
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
        scale = self.calc_scale_pinhole(depth)
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
                ray, ps=self.pixel_size, ks=ks, pointc=pointc_ideal, coherent=False
            )

        # Normalize to 1
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return psf

    def psf_map(
        self,
        depth=DEPTH,
        grid=7,
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
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # shape [grid**2, 1, ks, ks]

        psf_map = make_grid(psfs, nrow=grid, padding=0)[
            0, :, :
        ]  # shape [grid*ks, grid*ks]
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
        assert torch.get_default_dtype() == torch.float64 or torch.backends.mps.is_available(), (
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
        scale = self.calc_scale_ray(point[:, 2].item())
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

    def analysis_rms(self, depth=float("inf")):
        """Compute RMS-based error. Contain both RMS errors and RMS radius. This function needs more testing."""
        if depth == float("inf"):
            num_fields = 3
            fov_x = [0.0]
            fov_y = torch.linspace(0.0, self.hfov * 57.3, num_fields).tolist()

            all_rms_errors = []
            all_rms_radii = []
            for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
                # Ray tracing
                ray = self.sample_parallel(
                    fov_x=fov_x, fov_y=fov_y, num_rays=SPP_PSF, wvln=wvln
                )
                ray = self.trace2sensor(ray)

                # Green light point center for reference
                if i == 0:
                    pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(
                        -2
                    ) / ray.ra.sum(-1).add(EPSILON).unsqueeze(-1)
                    pointc_green = pointc_green.unsqueeze(-2).repeat(
                        1, 1, SPP_PSF, 1
                    )  # shape [1, num_fields, num_rays, 2]

                # Calculate RMS error for different FoVs
                o2_norm = (ray.o[..., :2] - pointc_green) * ray.ra.unsqueeze(-1)

                rms_error = ((o2_norm**2).sum(-1).sqrt() * ray.ra).sum(-1) / (
                    ray.ra.sum(-1) + EPSILON
                )  # shape [1, num_fields]
                rms_radius = (o2_norm**2).sum(-1).sqrt().max(dim=-1).values
                all_rms_errors.append(rms_error[0])
                all_rms_radii.append(rms_radius[0])

            # Calculate and print average across wavelengths
            avg_rms_error = torch.stack(all_rms_errors).mean(dim=0)
            avg_rms_radius = torch.stack(all_rms_radii).mean(dim=0)

            avg_rms_error = [round(value.item(), 3) for value in avg_rms_error]
            avg_rms_radius = [round(value.item(), 3) for value in avg_rms_radius]

            print(
                f"RMS average error (chief ray): center {avg_rms_error[0]} mm, middle {avg_rms_error[1]} mm, off-axis {avg_rms_error[-1]} mm"
            )
            print(
                f"RMS maximum radius (chief ray): center {avg_rms_radius[0]} mm, middle {avg_rms_radius[1]} mm, off-axis {avg_rms_radius[-1]} mm"
            )

        else:
            # Sample diagonal points
            grid = 20
            x = torch.linspace(0, 1, grid)
            y = torch.linspace(0, 1, grid)
            z = torch.full_like(x, depth)
            points = torch.stack((x, y, z), dim=-1)
            scale = self.calc_scale_ray(depth)

            # Ray position in the object space by perspective projection, because points are normalized
            point_obj_x = (
                points[..., 0] * scale * self.sensor_size[1] / 2
            )  # x coordinate
            point_obj_y = (
                points[..., 1] * scale * self.sensor_size[0] / 2
            )  # y coordinate
            point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)

            # Point center determined by green light
            ray = self.sample_from_points(
                points=point_obj, num_rays=SPP_PSF, wvln=DEFAULT_WAVE
            )
            ray = self.trace2sensor(ray)
            pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(
                -2
            ) / ray.ra.unsqueeze(-1).sum(-2).add(EPSILON)

            # Calculate RMS spot size
            rms = []
            for wvln in WAVE_RGB:
                # Trace rays to sensor plane
                ray = self.sample_from_points(
                    points=point_obj, num_rays=SPP_PSF, wvln=wvln
                )
                ray = self.trace2sensor(ray)

                # Calculate RMS error for different FoVs
                o2_norm = (
                    ray.o[..., :2] - pointc_green.unsqueeze(-2)
                ) * ray.ra.unsqueeze(-1)
                rms0 = torch.sqrt(
                    (o2_norm**2 * ray.ra.unsqueeze(-1)).sum((-2, -1))
                    / (ray.ra.sum(-1) + EPSILON)
                )
                rms.append(rms0)

            rms = torch.stack(rms, dim=0)
            rms = torch.mean(rms, dim=0)

            # Calculate RMS error for on-axis and off-axis
            rms_avg = round(rms.mean().item(), 3)
            rms_radius_on_axis = round(rms[0].item(), 3)  # Center point
            rms_radius_off_axis = round(rms[-1].item(), 3)  # Corner point

            print(
                f"RMS average error (chief ray): center {rms_radius_on_axis} mm, off-axis {rms_radius_off_axis} mm, average {rms_avg} mm"
            )

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

    def vignetting(self):
        """Compute vignetting."""
        pass

    def field_curvature(self):
        """Compute field curvature."""
        pass

    def ray_aberration(self):
        """Compute ray aberration."""
        pass

    def distortion(self, depth=DEPTH, grid_size=64):
        """Compute distortion map.

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

    def calc_foclen(self):
        """Calculate the focus length."""
        if (
            self.r_sensor < 8
        ):  # Cellphone lens, we usually use EFL to describe the lens.
            return self.calc_efl()
        else:  # Camera lens, we use the to describe the lens.
            return self.calc_bfl()

    @torch.no_grad()
    def calc_bfl(self, wvln=DEFAULT_WAVE):
        """Compute back focal length (BFL).

        BFL: Distance from the second principal point to focal plane.
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
        bfl = float(np.nanmean(bfl[ray.ra > 0].cpu().numpy()))

        return bfl

    def calc_efl(self):
        """Compute effective focal length (EFL). Effctive focal length is also commonly used to compute F/#.

        EFL: Defined by FoV and sensor radius.
        """
        return self.r_sensor / math.tan(self.hfov)

    def calc_eqfl(self):
        """35mm equivalent focal length. For cellphone lens, we usually use EFL to describe the lens.

        35mm sensor: 36mm * 24mm
        """
        return 21.63 / math.tan(self.hfov)

    @torch.no_grad()
    def calc_fnum(self):
        """Compute f-number."""
        _, pupilr = self.calc_entrance_pupil()
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
        focus_p = (ray.o[..., 2] - ray.d[..., 2] * t)[ray.ra > 0].cpu().numpy()
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
                points=torch.tensor([0, 0, depth]),
                num_rays=SPP_CALC,
                wvln=DEFAULT_WAVE,
            )
        ray, _ = self.trace(ray)

        # Calculate in-focus sensor position
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_p = ray.o[..., 2] - ray.d[..., 2] * t
        focus_p = focus_p[ray.ra > 0]
        focus_p = focus_p[~torch.isnan(focus_p) & (focus_p > 0)]
        infocus_sensor_d = torch.mean(focus_p)

        return infocus_sensor_d

    @torch.no_grad()
    def calc_hfov(self):
        """Compute half diagonal fov. Shot rays from edge of sensor, trace them to the object space and compute output angel as the fov."""
        # Sample rays going out from edge of sensor, shape [M, 3]
        o1 = torch.zeros([SPP_CALC, 3])
        o1 = torch.tensor([self.r_sensor, 0, self.d_sensor.item()]).repeat(SPP_CALC, 1)

        # Option 1: sample second points on exit pupil
        pupilz, pupilx = self.calc_exit_pupil()
        x2 = torch.linspace(-pupilx, pupilx, SPP_CALC)
        z2 = torch.full_like(x2, pupilz)
        # Option 2: sample second points on last surface
        # x2 = torch.linspace(0, self.surfaces[-1].r, SPP_CALC)
        # z2 = torch.full_like(x2, self.surfaces[-1].d.item())

        y2 = torch.full_like(x2, 0)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        ray = self.trace2obj(ray)

        # compute fov
        tan_fov = ray.d[..., 0] / ray.d[..., 2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))

        if torch.isnan(fov):
            fov = float(np.arctan(self.r_sensor / self.d_sensor.item()))
            print(f"Computing fov failed, set fov to {fov}.")
        else:
            fov = fov.item()

        return fov

    @torch.no_grad()
    def calc_principal(self):
        """Compute principal (front and back) planes."""
        # Backward ray tracing to compute first principal point
        ray = self.sample_parallel_2D(
            fov=0.0,
            depth=self.d_sensor.item(),
            num_rays=SPP_CALC,
            wvln=DEFAULT_WAVE,
            forward=False,
        )
        inc_ray = ray.clone()
        out_ray, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        front_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        # Forward ray tracing to compute second principal point
        ray = self.sample_parallel_2D(
            fov=0.0, depth=0.0, num_rays=SPP_CALC, wvln=DEFAULT_WAVE, forward=True
        )
        inc_ray = ray.clone()
        out_ray, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        back_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        return front_principal, back_principal

    @torch.no_grad()
    def calc_scale(self, depth, method="pinhole"):
        """Calculate the scale factor."""
        if method == "pinhole":
            scale = self.calc_scale_pinhole(depth)
        elif method == "raytracing":
            scale = self.calc_scale_ray(depth)
        else:
            raise ValueError(f"Invalid method: {method}.")

        return scale

    @torch.no_grad()
    def calc_scale_pinhole(self, depth):
        """Assume the first principle point is at (0, 0, 0), use pinhole camera model to calculate the scale factor.

        Note: due to distortion, the scale factor computed here is larger than the actual scale factor.

        Args:
            depth (float): depth of the object.

        Returns:
            scale (float): scale factor.
        """
        scale = -depth * float(np.tan(self.hfov)) / self.r_sensor
        return scale

    @torch.no_grad()
    def calc_scale_ray(self, depth):
        """Use ray tracing to compute scale factor."""
        if isinstance(depth, float) or isinstance(depth, int):
            # Sample rays [num_grid, num_grid, spp, 3] from the object plane
            ray = self.sample_point_source(depth=depth, num_rays=SPP_CALC, num_grid=64)

            # Map points from object space to sensor space, ground-truth
            o1 = ray.o.clone()[..., :2]
            o1 = torch.flip(o1, [0, 1])

            ray, _ = self.trace(ray)
            o2 = ray.project_to(self.d_sensor)  # shape [num_grid, num_grid, spp, 2]

            # Use only center region of points, because we assume center points have no distortion
            center_start = GEO_GRID // 2 - GEO_GRID // 8
            center_end = GEO_GRID // 2 + GEO_GRID // 8
            o1_center = o1[center_start:center_end, center_start:center_end, :, :]
            o2_center = o2[center_start:center_end, center_start:center_end, :, :]
            ra_center = ray.ra.clone().detach()[
                center_start:center_end, center_start:center_end, :
            ]

            x1 = o1_center[:, :, 0, 0]  # shape [num_grid // 4, num_grid // 4]
            x2 = (o2_center[:, :, :, 0] * ra_center).sum(dim=-1) / (ra_center).sum(
                dim=-1
            ).add(EPSILON)

            # Calculate scale factor (currently assume rotationally symmetric)
            scale_x = x1 / x2  # shape [num_grid // 4, num_grid // 4]
            try:
                scale = torch.mean(scale_x[~scale_x.isnan()]).item()
            except Exception as e:
                print(f"Error calculating scale: {e}")
                scale = -depth * np.tan(self.hfov) / self.r_sensor
            return scale

        elif isinstance(depth, torch.Tensor) and len(depth.shape) == 1:
            scale = []
            for d in depth:
                scale.append(self.calc_scale_ray(d.item()))
            scale = torch.tensor(scale)
            return scale

        else:
            raise ValueError("Invalid depth type.")

    @torch.no_grad()
    def chief_ray(self):
        """Compute chief ray. We can use chief ray for fov, magnification.
        Chief ray, a ray goes through center of aperture.
        """
        # sample rays with shape [SPP_CALC, 3]
        pupilz, pupilx = self.calc_exit_pupil()
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
    def calc_exit_pupil(self, shrink_pupil=False):
        """Sample **forward** rays to compute z coordinate and radius of exit pupil.

        Args:
            shrink_pupil (bool): whether to shrink the pupil.

        Reference:
            [1] Exit pupil: how many rays can come from sensor to object space.
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        return self.calc_entrance_pupil(entrance=False, shrink_pupil=shrink_pupil)

    @torch.no_grad()
    def calc_entrance_pupil(self, entrance=True, shrink_pupil=False):
        """Sample **backward** rays, return z coordinate and radius of entrance pupil.

        Args:
            entrance (bool): whether to compute entrance pupil.
            shrink_pupil (bool): whether to shrink the pupil.

        Reference:
            [1] Entrance pupil: how many rays can come from object space to sensor.
            [2] https://en.wikipedia.org/wiki/Entrance_pupil: "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
        """
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            if entrance:
                return self.surfaces[0].d.item(), self.surfaces[0].r
            else:
                return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # Sample M rays from edge of aperture to last surface.
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r
        ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)

        # Sample phi ranges from [-0.5rad, 0.5rad]
        phi = torch.linspace(-0.5, 0.5, SPP_CALC)
        if entrance:
            d = torch.stack(
                (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
            )
        else:
            d = torch.stack(
                (torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)), axis=-1
            )

        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing
        if entrance:
            # Trace from aperture edge to first surface
            lens_range = range(0, self.aper_idx)
            ray, _ = self.trace(ray, lens_range=lens_range)
        else:
            # Trace from aperture edge to last surface
            lens_range = range(self.aper_idx + 1, len(self.surfaces))
            ray, _ = self.trace(ray, lens_range=lens_range)

        # Compute intersection points. o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.ra != 0][:, 0], ray.o[ray.ra != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.ra != 0][:, 0], ray.d[ray.ra != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)
        if len(intersection_points) == 0:
            print("No intersection points found, use the first surface as pupil.")
            if entrance:
                avg_pupilz = self.surfaces[0].d.item()
                avg_pupilx = self.surfaces[0].r
            else:
                avg_pupilz = self.surfaces[-1].d.item()
                avg_pupilx = self.surfaces[-1].r
        else:
            avg_pupilx = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

        if avg_pupilx < EPSILON:
            print("Small pupil is detected, use the first surface as pupil.")
            if entrance:
                avg_pupilz = self.surfaces[0].d.item()
                avg_pupilx = self.surfaces[0].r
            else:
                avg_pupilz = self.surfaces[-1].d.item()
                avg_pupilx = self.surfaces[-1].r

        if shrink_pupil:
            avg_pupilx *= 0.5
        return avg_pupilz, avg_pupilx

    @staticmethod
    def filter_parallel_lines(Di, Dj, threshold=1e-3):
        """
        Identify line pairs that are nearly parallel by computing the cosine of the angle
        between their directions. If |cos(theta)| ~ 1, they're parallel or anti-parallel.

        Args:
            Di (torch.Tensor): Direction vectors for line i, shape [M, 2].
        Dj (torch.Tensor): Direction vectors for line j, shape [M, 2].
        threshold (float): If 1 - |cos(theta)| < threshold, we consider lines nearly parallel.

        Returns:
            torch.BoolTensor: A mask of shape [M], where True means "not parallel."
        """
        # Dot product for each pair
        dot_ij = (Di * Dj).sum(dim=-1)  # shape [M]
        # Norms for each pair
        norm_i = Di.norm(dim=-1)
        norm_j = Dj.norm(dim=-1)

        # Avoid division by zero
        denom = (norm_i * norm_j).clamp_min(1e-12)

        # cos(theta) for each pair
        cos_theta = dot_ij / denom  # shape [M]

        # Lines are "near parallel" if |cos(theta)| is extremely close to 1
        # i.e., if 1 - |cos(theta)| < threshold
        parallel_mask = (1.0 - cos_theta.abs()) < threshold

        # We return the inverse: True means "keep" (not parallel)
        return ~parallel_mask

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
        if N < 2:
            # Not enough lines to intersect
            return torch.empty(0, 2, device=origins.device)

        # Create pairwise combinations of indices
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # Shape: [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # Shape: [N*(N-1)/2, 2]
        Di = directions[idx_i]  # Shape: [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # Shape: [N*(N-1)/2, 2]

        # 1) Filter out nearly parallel lines
        not_parallel_mask = GeoLens.filter_parallel_lines(Di, Dj, threshold=1e-3)

        # Keep only the pairs that aren't parallel
        Oi = Oi[not_parallel_mask]
        Oj = Oj[not_parallel_mask]
        Di = Di[not_parallel_mask]
        Dj = Dj[not_parallel_mask]

        # If everything got filtered out, return empty
        if Oi.shape[0] == 0:
            return torch.empty(0, 2, device=origins.device)

        # 2) Construct the system A x = b
        # Vector from Oi to Oj
        b = Oj - Oi  # Shape: [N*(N-1)/2, 2]

        # Coefficients matrix A
        A = torch.stack([Di, -Dj], dim=-1)  # Shape: [N*(N-1)/2, 2, 2]

        # Solve the linear system Ax = b
        # Using least squares to handle the case of no exact solution
        x, _ = torch.linalg.lstsq(
            A,
            b.unsqueeze(-1),
            driver='gelsd'
        )[:2]
        if x.dim() == 2:  # If only one system was solved, add a batch dimension.
            x = x.unsqueeze(0)
        x = x.squeeze(-1)  # Shape: [N*(N-1)/2, 2]
        s = x[:, 0]
        t = x[:, 1]

        # Calculate the intersection points using either rays
        P_i = Oi + s.unsqueeze(-1) * Di  # Shape: [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # Shape: [N*(N-1)/2, 2]

        # Take the average to mitigate numerical precision issues
        P = (P_i + P_j) / 2

        return P

    def sample_point_source_with_offset(self, depth, num_rays, num_grid, wvln, offset):
        """
        Wraps self.sample_point_source to apply a manual offset to the ray source.
        
        Args:
            depth: The object depth.
            num_rays: Number of rays to sample.
            num_grid: Grid resolution for sampling.
            wvln: Wavelength.
            offset: A tuple (dx, dy) specifying the offset in object space.
        
        Returns:
            A list or batch of rays with modified source positions.
        """
        ray = self.sample_point_source(depth=depth, num_rays=num_rays, num_grid=num_grid, wvln=wvln, importance_sampling=True)
    
        # Apply offset to ray's origin (or object coordinates).
        ray.o[..., :2] += torch.tensor(offset, device=self.device)
        return ray

    def get_cached_rays(self, depth, wvln, offset, num_grid=15):
        """
        Retrieve rays sampled at a given depth and wavelength. If not already cached,
        call sample_point_source once and store the result.
        
        Args:
            depth: The object depth.
            wvln: Wavelength.
            offset: A direction specifying the offset in object space. 'c', 'h' or 'v'
        
        Returns:
            A list or batch of rays with modified source positions.
        """
        # Create a key based on depth and wavelength
        key = (depth, wvln, offset, num_grid)
        if not hasattr(self, "_cached_rays"):
            self._cached_rays = {}
        if key not in self._cached_rays:
            # Sample rays once and store them
            if offset == 'c':
                self._cached_rays[key] = self.sample_point_source(
                    depth=depth,
                    num_rays=1024,
                    num_grid=num_grid,
                    wvln=wvln
                )
            elif offset == 'h':
                self._cached_rays[key] = self.sample_point_source_with_offset(
                    depth=depth,
                    num_rays=1024,
                    num_grid=num_grid,
                    wvln=wvln,
                    offset=(1, 0.0)
                )
            elif offset == 'v':
                self._cached_rays[key] = self.sample_point_source_with_offset(
                    depth=depth,
                    num_rays=1024,
                    num_grid=num_grid,
                    wvln=wvln,
                    offset=(0.0, 1)
                )
        return self._cached_rays[key].clone()

    def compute_effective_horizontal_and_vertical_magnification(self, depth):
        """
        Compute the effective horizontal and vertical magnification of the full lens system.
        This is done by sampling rays from a point source with and without a small
        horizontal and vertical offset and then comparing the sensor positions.
    
        Args:
            depth (float): The object depth for ray sampling.
    
        Returns:
            torch.Tensor: The computed horizontal and vertical magnification.
        """
        delta = 1  # Small horizontal offset in object space.
        M_x_list = []
        M_y_list = []
        for wvln in WAVE_RGB:
            # Sample a central ray (no offset)
            ray_central = self.get_cached_rays(
                depth=depth,
                wvln=wvln,
                offset='c'
            )
            ray_central, _ = self.trace(ray_central)
            sensor_central = ray_central.project_to(self.d_sensor)

            # Sample a ray with a small horizontal offset (delta, 0)
            ray_offset_h = self.get_cached_rays(
                depth=depth,
                wvln=wvln,
                offset='h'
            )
            ray_offset_h, _ = self.trace(ray_offset_h)
            sensor_offset_h = ray_offset_h.project_to(self.d_sensor)

            # Sample a ray with a small vertical offset (0, delta)
            ray_offset_v = self.get_cached_rays(
                depth=depth,
                wvln=wvln,
                offset='v'
            )
            ray_offset_v, _ = self.trace(ray_offset_v)
            sensor_offset_v = ray_offset_v.project_to(self.d_sensor)

            # Compute horizontal magnification: (difference in sensor x) / delta.
            M_x = torch.abs((sensor_offset_h[..., 0] - sensor_central[..., 0]) / delta)
            # Compute vertical magnification: (difference in sensor y) / delta.
            M_y = torch.abs((sensor_offset_v[..., 1] - sensor_central[..., 1]) / delta)
            M_x_list.append(M_x)
            M_y_list.append(M_y)
        return torch.mean(torch.stack(M_x_list)), torch.mean(torch.stack(M_y_list))

    # ====================================================================================
    # Lens operation
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, depth=float("inf")):
        """Refocus the lens to a depth distance by changing sensor position.

        Args:
            depth (float): depth distance.

        Note:
            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Calculate in-focus sensor position
        d_sensor_new = self.calc_foc_plane(depth=depth)

        # Update sensor position
        assert d_sensor_new > 0, "sensor position is negative."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def set_aperture(self, fnum=None, foclen=None, aper_r=None):
        """Change aperture radius.

        TODO: This function will be deprecated in the future.
        """
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
        """Set F number and aperture radius using binary search."""
        target_pupil_r = self.foclen / fnum / 2

        # Binary search to find aperture radius that gives desired exit pupil radius
        optim_aper_r = target_pupil_r
        aper_r_min = 0.5 * target_pupil_r  # Start with small radius
        aper_r_max = 2.0 * target_pupil_r  # Start with large radius

        for _ in range(8):
            self.surfaces[self.aper_idx].r = optim_aper_r
            _, pupilr = self.calc_entrance_pupil()

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
    def set_target_fov_fnum(self, hfov, fnum, imgh=None):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        TODO: This function will be deprecated in the future.
        """
        if imgh is not None:
            self.r_sensor = imgh / 2
        self.hfov = hfov
        self.fnum = fnum

        self.foclen = self.calc_efl()
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].r = float(aper_r)

    @torch.no_grad()
    def set_fov(self, hfov):
        """Set FoV. This function is used to assign design targets.

        Args:
            hfov (float): half diagonal-FoV in degree.
        """
        self.hfov = hfov

    @torch.no_grad()
    def set_sensor(self, sensor_res=None, sensor_size=None):
        """Set camera sensor, define pixel size and sensor size.

        Args:
            sensor_res (list): Resolution, pixel number.
            sensor_size (list): Sensor size in [mm].
        """
        assert not (sensor_res is None and sensor_size is None), (
            "Cannot set both sensor_res and sensor_size"
        )

        if sensor_res is not None:
            # Change sensor size, resolution and pixel size. Do not change sensor diagonal radius.
            if isinstance(sensor_res, int):
                sensor_res = (sensor_res, sensor_res)
            self.sensor_res = sensor_res
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

            self.post_computation()

        else:
            # Change sensor size, resolution and sensor diagonal radius. Do not change pixel size.
            self.sensor_size = sensor_size
            self.sensor_res = [
                sensor_size[0] / self.pixel_size,
                sensor_size[1] / self.pixel_size,
            ]
            self.r_sensor = math.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2) / 2

            self.post_computation()

    @torch.no_grad()
    def prune_surf(self, expand_surf=None, surface_range=None):
        """Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            expand_surf (float): extra height to reserve.
                - For cellphone lens, we usually use 0.1mm or 0.05 * r_sensor.
                - For camera lens, we usually use 0.5mm or 0.1 * r_sensor.
            surface_range (list): surface range to prune.
        """
        # Settings
        surface_range = (
            self.find_diff_surf() if surface_range is None else surface_range
        )
        if self.is_cellphone:
            expand_surf = 0.05 if expand_surf is None else expand_surf

            # Reset lens to maximum height(sensor radius)
            for i in surface_range:
                # self.surfaces[i].r = self.r_sensor
                self.surfaces[i].r = max(self.r_sensor, self.surfaces[self.aper_idx].r)
        else:
            expand_surf = 0.2 if expand_surf is None else expand_surf

        # Sample full-fov rays to compute valid surface height
        if self.hfov is not None:
            fov = self.hfov
        else:
            fov = float(np.arctan(self.r_sensor / self.d_sensor.item()))

        ray = self.sample_parallel_2D(
            fov=fov * 57.3, num_rays=GEO_GRID, entrance_pupil=True
        )

        ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
        ray_o_record = torch.stack(
            ray_o_record, dim=-2
        )  # [num_rays, num_surfaces + 2, 3]
        ray_x_record = ray_o_record[:, :, 0]
        for i in surface_range:
            # Filter out nan values and compute the maximum height
            valid_heights = ray_x_record[:, i + 1].abs()
            valid_heights = valid_heights[~torch.isnan(valid_heights)]
            if len(valid_heights) > 0:
                 max_ray_height = valid_heights.max().item()
            else:
                print(f"No valid rays for surface {i}, use the maximum height of the surface.")
                max_ray_height = self.surfaces[i].r

            if max_ray_height > 0:
                max_height_value_range = self.surfaces[i].max_height()
                self.surfaces[i].r = min(
                    max_ray_height * (1 + expand_surf), max_height_value_range
                )
            else:
                print("Get max height failed, use the maximum height of the surface.")

    @torch.no_grad()
    def correct_shape(self, expand_surf=None):
        """Correct wrong lens shape during the lens design."""
        aper_idx = self.aper_idx
        optim_surf_range = self.find_diff_surf()
        shape_changed = False

        # Rule 1: Move the first surface to z = 0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # Rule 2: Move lens group to get a fixed aperture distance. Only for aperture at the first surface.
        if aper_idx == 0:
            d_aper = 0.1 if self.is_cellphone else 2.0

            # If the first surface is concave, use the maximum negative sag.
            aper_r = self.surfaces[aper_idx].r
            # sag1 = -self.surfaces[aper_idx + 1].surface(aper_r, 0).item()
            sag1 = -self.surfaces[aper_idx + 1].sag(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in optim_surf_range:
                self.surfaces[i].d -= delta_aper

        # Rule 3: If two surfaces overlap (at center), seperate them by a small distance
        for i in range(0, len(self.surfaces) - 1):
            if self.surfaces[i].d > self.surfaces[i + 1].d:
                self.surfaces[i + 1].d += 0.1
                shape_changed = True

        # Rule 4: Prune all surfaces
        self.prune_surf(expand_surf=expand_surf)

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
        multi_plot=False,
        zmx_format=True,
        depth=float("inf"),
        render=False,
        render_unwarp=False,
        lens_title=None,
    ):
        """Analyze the optical lens.

        Args:
            save_name (str): save name.
            multi_plot (bool): plot RGB seperately.
            plot_invalid (bool): plot invalid rays.
            zmx_format (bool): plot in Zemax format.
            depth (float): object depth distance.
            render (bool): whether render an image.
            render_unwarp (bool): whether unwarp the rendered image.
            lens_title (str): lens title
        """

        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            multi_plot=multi_plot,
            entrance_pupil=True,
            zmx_format=zmx_format,
            lens_title=lens_title,
            depth=depth,
        )

        # Draw spot diagram and PSF map
        # self.draw_psf_map(save_name=save_name, ks=101, depth=depth)

        # Calculate RMS error
        self.analysis_rms(depth=depth)

        # Render an image, compute PSNR and SSIM
        if render:
            if depth == float("inf"):
                depth = DEPTH
            img_org = cv.cvtColor(cv.imread("./datasets/IQ/img1.png"), cv.COLOR_BGR2RGB)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                noise=0.01,
            )

    @torch.no_grad()
    def draw_layout(
        self,
        filename,
        depth=float("inf"),
        entrance_pupil=True,
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
    ):
        """Plot lens layout with ray tracing."""
        num_rays = 11
        num_views = 3

        # Lens title
        if lens_title is None:
            if self.aper_idx is not None:
                fnum = self.foclen / self.calc_entrance_pupil()[1] / 2
                lens_title = f"FoV{round(2 * self.hfov * 57.3, 1)}({int(self.calc_eqfl())}mm EFL)_F/{round(fnum, 2)}_DIAG{round(self.r_sensor * 2, 2)}mm_FocLen{round(self.foclen, 2)}mm"
            else:
                lens_title = f"FoV{round(2 * self.hfov * 57.3, 1)}({int(self.calc_eqfl())}mm EFL)_DIAG{round(self.r_sensor * 2, 2)}mm_FocLen{round(self.foclen, 2)}mm"

        # Draw lens layout
        if not multi_plot:
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            views = np.linspace(0, float(np.rad2deg(self.hfov) * 0.99), num=num_views)
            ax, fig = self.draw_setup_2d(zmx_format=zmx_format)

            for i, view in enumerate(views):
                # Sample rays, shape (num_view, num_rays, 3)
                if depth == float("inf"):
                    ray = self.sample_parallel_2D(
                        fov=view,
                        wvln=WAVE_RGB[2 - i],
                        num_rays=num_rays,
                        entrance_pupil=entrance_pupil,
                    )  # shape (num_rays, 3)
                else:
                    ray = self.sample_point_source_2D(
                        fov=view,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=WAVE_RGB[2 - i],
                        entrance_pupil=entrance_pupil,
                    )  # shape (num_rays, 3)

                # Trace rays to sensor and plot ray paths
                _, ray_o_record = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.draw_raytraces_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )

            ax.axis("off")
            ax.set_title(lens_title, fontsize=10)
            if filename.endswith(".png"):
                fig.savefig(filename, format="png", dpi=600)
            else:
                raise ValueError("Invalid file extension")
            plt.close()

        else:
            views = np.linspace(0, np.rad2deg(self.hfov) * 0.99, num=num_views)
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(lens_title)

            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.draw_setup_2d(ax=ax, fig=fig, zmx_format=zmx_format)

                for view in views:
                    if depth == float("inf"):
                        ray = self.sample_parallel_2D(
                            fov=view,
                            num_rays=num_rays,
                            wvln=wvln,
                            entrance_pupil=entrance_pupil,
                        )  # shape (num_rays, 3)
                    else:
                        ray = self.sample_point_source_2D(
                            fov=view,
                            depth=depth,
                            num_rays=num_rays,
                            wvln=wvln,
                            entrance_pupil=entrance_pupil,
                        )  # shape (num_rays, 3)

                    ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.draw_raytraces_2d(
                        ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                    )
                    ax.axis("off")

            if filename.endswith(".png"):
                fig.savefig(filename, format="png", dpi=300)
            else:
                raise ValueError("Invalid file extension")
            plt.close()

    def draw_layout_3d(self, filename=None):
        """Draw 3D layout of the lens system.

        Args:
            filename (str, optional): Path to save the figure. Defaults to None.

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        figsize = (10, 6)
        view_angle = 30
        show = True

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Enable depth sorting for proper occlusion
        ax.set_proj_type(
            "persp"
        )  # Use perspective projection for better depth perception

        # Draw each surface
        for i, surf in enumerate(self.surfaces):
            surf.draw_widget3D(ax)

            # Connect current surface with previous surface if material is not air
            if i > 0 and self.surfaces[i - 1].mat2.get_name() != "air":
                # Get edge points of current and previous surfaces
                theta = np.linspace(0, 2 * np.pi, 256)

                # Current surface edge
                curr_edge_x = surf.r * np.cos(theta)
                curr_edge_y = surf.r * np.sin(theta)
                curr_edge_z = np.array(
                    [
                        surf.surface_with_offset(
                            torch.tensor(curr_edge_x[j], device=surf.device),
                            torch.tensor(curr_edge_y[j], device=surf.device),
                        ).item()
                        for j in range(len(theta))
                    ]
                )

                # Previous surface edge
                prev_surf = self.surfaces[i - 1]
                prev_edge_x = prev_surf.r * np.cos(theta)
                prev_edge_y = prev_surf.r * np.sin(theta)
                prev_edge_z = np.array(
                    [
                        prev_surf.surface_with_offset(
                            torch.tensor(prev_edge_x[j], device=prev_surf.device),
                            torch.tensor(prev_edge_y[j], device=prev_surf.device),
                        ).item()
                        for j in range(len(theta))
                    ]
                )

                # Create a cylindrical surface connecting the two edges
                theta_mesh, t_mesh = np.meshgrid(theta, np.array([0, 1]))

                # Interpolate between previous and current surface edges
                x_mesh = (
                    prev_edge_x[None, :] * (1 - t_mesh) + curr_edge_x[None, :] * t_mesh
                )
                y_mesh = (
                    prev_edge_y[None, :] * (1 - t_mesh) + curr_edge_y[None, :] * t_mesh
                )
                z_mesh = (
                    prev_edge_z[None, :] * (1 - t_mesh) + curr_edge_z[None, :] * t_mesh
                )

                # Plot the connecting surface with sort_zpos for proper occlusion
                surf = ax.plot_surface(
                    z_mesh,
                    x_mesh,
                    y_mesh,
                    color="lightblue",
                    alpha=0.3,
                    edgecolor="lightblue",
                    linewidth=0.5,
                    antialiased=True,
                )
                # Set the zorder based on the mean z position for better occlusion
                surf._sort_zpos = np.mean(z_mesh)

        # Draw sensor as a rectangle
        if hasattr(self, "sensor_size") and hasattr(self, "d_sensor"):
            # Get sensor dimensions
            sensor_width = self.sensor_size[0]
            sensor_height = self.sensor_size[1]
            sensor_z = self.d_sensor.item()

            # Create sensor vertices
            half_width = sensor_width / 2
            half_height = sensor_height / 2

            # Define the corners of the rectangle
            x = np.array(
                [-half_width, half_width, half_width, -half_width, -half_width]
            )
            y = np.array(
                [-half_height, -half_height, half_height, half_height, -half_height]
            )
            z = np.full_like(x, sensor_z)

            # Plot the sensor rectangle
            ax.plot(z, x, y, color="black", linewidth=1.5)

            # Add a semi-transparent surface for the sensor
            sensor_x, sensor_y = np.meshgrid(
                np.linspace(-half_width, half_width, 2),
                np.linspace(-half_height, half_height, 2),
            )
            sensor_z = np.full_like(sensor_x, sensor_z)
            sensor_surf = ax.plot_surface(
                sensor_z,
                sensor_x,
                sensor_y,
                color="gray",
                alpha=0.3,
                edgecolor="black",
                linewidth=0.5,
            )
            # Set the zorder for the sensor
            sensor_surf._sort_zpos = sensor_z.mean()

        # Set axis properties
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        ax.view_init(elev=20, azim=-view_angle - 90)

        # Make all axes have the same scale (unit step size)
        ax.set_box_aspect([1, 1, 1])
        ax.set_aspect("equal")

        # Enable depth sorting for proper occlusion
        from matplotlib.collections import PathCollection

        for c in ax.collections:
            if isinstance(c, PathCollection):
                c.set_sort_zpos(c.get_offsets()[:, 2].mean())

        plt.tight_layout()

        if filename:
            fig.savefig(f"{filename}.png", format="png", dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def draw_raytraces_2d(self, ray_o_record, ax, fig, color="b"):
        """Plot ray paths.

        Args:
            ray_o_record (list): list of intersection points.
            ax (matplotlib.axes.Axes): matplotlib axes.
            fig (matplotlib.figure.Figure): matplotlib figure.
        """
        # shape (num_view, num_rays, num_path, 2)
        ray_o_record = torch.stack(ray_o_record, dim=-2).cpu().numpy()
        if ray_o_record.ndim == 3:
            ray_o_record = ray_o_record[None, ...]

        for idx_view in range(ray_o_record.shape[0]):
            for idx_ray in range(ray_o_record.shape[1]):
                ax.plot(
                    ray_o_record[idx_view, idx_ray, :, 2],
                    ray_o_record[idx_view, idx_ray, :, 0],
                    color,
                    linewidth=0.8,
                )

                # ax.scatter(
                #     ray_o_record[idx_view, idx_ray, :, 2],
                #     ray_o_record[idx_view, idx_ray, :, 0],
                #     "b",
                #     marker="x",
                # )

        return ax, fig

    def draw_setup_2d(
        self,
        ax=None,
        fig=None,
        color="k",
        linestyle="-",
        zmx_format=False,
        fix_bound=False,
    ):
        """Draw lens layout in a 2D plot."""

        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        # Draw lens surfaces
        for i, s in enumerate(self.surfaces):
            s.draw_widget(ax)

        # Connect two surfaces
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.n > 1.1:
                s_prev = self.surfaces[i]
                s = self.surfaces[i + 1]

                r_prev = float(s_prev.r)
                r = float(s.r)
                sag_prev = s_prev.surface_with_offset(r_prev, 0.0).item()
                sag = s.surface_with_offset(r, 0.0).item()

                if zmx_format:
                    if r > r_prev:
                        z = np.array([sag_prev, sag_prev, sag])
                        x = np.array([r_prev, r, r])
                    else:
                        z = np.array([sag_prev, sag, sag])
                        x = np.array([r_prev, r, r])
                else:
                    z = np.array([sag_prev, sag])
                    x = np.array([r_prev, r])

                ax.plot(z, -x, color, linewidth=0.75)
                ax.plot(z, x, color, linewidth=0.75)
                s_prev = s

        # Draw sensor
        ax.plot(
            [self.d_sensor.item(), self.d_sensor.item()],
            [-self.r_sensor, self.r_sensor],
            color,
        )

        # Figure size
        if fix_bound:
            ax.set_aspect("equal")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect("equal", adjustable="datalim", anchor="C")
            ax.minorticks_on()
            ax.set_xlim(-0.5, 7.5)
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig

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

    @torch.no_grad()
    def draw_spot_diagram(self, M=5, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """Draw spot diagram of the lens. Shot rays from grid points in object space, trace to sensor and visualize."""
        # Sample and trace rays from grid points
        ray = self.sample_point_source(
            depth=depth,
            num_rays=SPP_CALC,
            num_grid=GEO_GRID,
            wvln=wvln,
        )
        ray = self.trace2sensor(ray)
        o2 = -ray.o.clone().cpu().numpy()
        ra = ray.ra.clone().cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(M, M, figsize=(30, 30))
        for i in range(M):
            for j in range(M):
                ra_ = ra[:, i, j]
                x, y = o2[:, i, j, 0], o2[:, i, j, 1]
                x, y = x[ra_ > 0], y[ra_ > 0]
                xc, yc = x.sum() / ra_.sum(), y.sum() / ra_.sum()

                # scatter plot
                axs[i, j].scatter(x, y, 1, "black")
                axs[i, j].scatter([xc], [yc], None, "r", "x")
                axs[i, j].set_aspect("equal", adjustable="datalim")

        if save_name is None:
            save_name = f"./spot{-depth}mm.png"
        else:
            save_name = f"{save_name}_spot{-depth}mm.png"

        plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close()

    @torch.no_grad()
    def draw_spot_radial(self, M=3, depth=DEPTH, save_name=None):
        """Draw radial spot diagram of the lens.

        Args:
            M (int, optional): field number. Defaults to 3.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample and trace rays
        # mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(
            depth=depth,
            num_rays=1024,
            num_grid=M * 2 - 1,
            wvln=DEFAULT_WAVE,
        )
        ray, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        o2 = torch.flip(ray.o.clone(), [1, 2]).cpu().numpy()
        ra = torch.flip(ray.ra.clone(), [1, 2]).cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(1, M, figsize=(M * 12, 10))
        for i in range(M):
            i_bias = i + M - 1

            # calculate center of mass
            ra_ = ra[:, i_bias, i_bias]
            x, y = o2[:, i_bias, i_bias, 0], o2[:, i_bias, i_bias, 1]
            x, y = x[ra_ > 0], y[ra_ > 0]
            xc, yc = x.sum() / ra_.sum(), y.sum() / ra_.sum()

            # scatter plot
            axs[i].scatter(x, y, 12, "black")
            axs[i].scatter([xc], [yc], 400, "r", "x")

            # visualization
            axs[i].set_aspect("equal", adjustable="datalim")
            axs[i].tick_params(axis="both", which="major", labelsize=18)
            axs[i].spines["top"].set_linewidth(4)
            axs[i].spines["bottom"].set_linewidth(4)
            axs[i].spines["left"].set_linewidth(4)
            axs[i].spines["right"].set_linewidth(4)

        # Save figure
        if save_name is None:
            plt.savefig(
                f"./spot{-depth}mm_radial.svg",
                bbox_inches="tight",
                format="svg",
                dpi=1200,
            )
        else:
            plt.savefig(
                f"{save_name}_spot{-depth}mm_radial.svg",
                bbox_inches="tight",
                format="svg",
                dpi=1200,
            )

        plt.close()

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

                fov_deg = round(fov * self.hfov * 57.3, 1)
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

    # ====================================================================================
    # Loss functions and lens design constraints
    # ====================================================================================
    def init_constraints(self):
        """Initialize constraints for the lens design."""
        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.dist_min = 0.1
            self.dist_max = 0.6  # float("inf")
            self.thickness_min = 0.3
            self.thickness_max = 1.5
            self.flange_min = 0.5
            self.flange_max = float("inf")

            self.sag_max = 0.8
            self.grad_max = 1.0
            self.grad2_max = 100.0
        else:
            self.is_cellphone = False

            self.dist_min = 2.5
            self.dist_max = 25.0 # float("inf")
            self.thickness_min = 3.0
            self.thickness_max = 40.0 # float("inf")
            self.flange_min = 29.6
            self.flange_max = 29.8 # float("inf")

            self.sag_max = 8.0
            self.grad_max = 1.0
            self.grad2_max = 100.0

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
            print(f"Losses in loss_reg: {loss_focus}, {loss_intersec}, {loss_surf}, {loss_angle}")

        return loss_reg

    def loss_anamorphic(self, depth, target_local_squeeze, target_global_squeeze):
        """
        Compute a combined loss for anamorphic surfaces that includes:
          - Local penalties on sag, 1st-, and 2nd-order derivatives.
          - A local squeeze penalty for each anamorphic surface.
          - A global squeeze penalty computed from ray-traced effective magnification.
    
        Args:
            depth (float): The object depth from which rays are sampled.
            target_local_squeeze (float): Target local squeeze value.
            target_global_squeeze (float): Target global squeeze value.
    
        Returns:
            torch.Tensor: Scalar loss value.
        """
        total_loss = torch.tensor([0.0], device=self.device)

        # Local loss terms for each anamorphic surface.
        for idx in self.find_diff_surf():
            surface = self.surfaces[idx]
            if not isinstance(surface, Anamorphic):
                continue
            # Sample 20 points along the x-axis (from 0 to the aperture radius)
            x_vals = torch.linspace(0.0, 1.0, 20).to(self.device) * surface.r
            y_vals = torch.zeros_like(x_vals)

            # --- Sag Penalty ---
            sag_vals = surface.sag(x_vals, y_vals)
            # Only penalize if the sag variation exceeds sag_max.
            sag_penalty = torch.nn.functional.relu((sag_vals.max() - sag_vals.min()) - self.sag_max)
            total_loss += sag_penalty

            # --- First-order Derivative Penalty ---
            grad_vals = surface.dfdxyz(x_vals, y_vals)[0]
            grad_penalty = 10.0 * torch.nn.functional.relu(grad_vals.abs().max() - self.grad_max)
            total_loss += grad_penalty

            # --- Second-order Derivative Penalty ---
            grad2_vals = surface.d2fdxyz2(x_vals, y_vals)[0]
            grad2_penalty = 10.0 * torch.nn.functional.relu(grad2_vals.abs().max() - self.grad2_max)
            total_loss += grad2_penalty
            print(f'Total loss: {total_loss}')

        # Global Squeeze Constraint via ray tracing.
        horiz_mag, vert_mag = self.compute_effective_horizontal_and_vertical_magnification(depth)
        print(f"Horizontal Magnification: {horiz_mag}")
        print(f"Vertical Magnification: {vert_mag}")
        global_squeeze = horiz_mag / (vert_mag + 1e-8)
        print(f"Global Squeeze: {global_squeeze}")
        global_squeeze_weight = 1.0 # keeping this big for now - this is a strong constraint
        global_squeeze_penalty = global_squeeze_weight * (global_squeeze - target_global_squeeze)**2
        print(f"Global Squeeze Penalty: {global_squeeze_penalty}")
        total_loss += global_squeeze_penalty

        return total_loss.item()

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

    def loss_rms(self, depth=DEPTH):
        """Compute RGB RMS error per pixel, forward rms error.

        Can also revise this function to plot PSF.
        """
        # PSF and RMS by patch
        rms = 0.0
        for wvln in WAVE_RGB:
            ray = self.sample_point_source(
                depth=depth,
                num_rays=SPP_PSF,
                num_grid=GEO_GRID,
                wvln=wvln,
            )
            ray, _ = self.trace(ray)
            o2 = ray.project_to(self.d_sensor)
            o2_center = (o2 * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(
                EPSILON
            ).unsqueeze(-1)
            # normalized to center (0, 0)
            o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)
            rms += torch.sum(o2_norm**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra)

        return rms / 3

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

    # ====================================================================================
    # Optimization
    # ====================================================================================
    def get_optimizer_params(
        self,
        lr=[1e-4, 1e-4, 1e-1, 1e-3],
        decay=0.01,
        optim_surf_range=None,
        optim_mat=False,
    ):
        """Get optimizer parameters for different lens surface.

        Recommendation:
            For cellphone lens: [c, d, k, a], [1e-4, 1e-4, 1e-1, 1e-4]
            For camera lens: [c, d, 0, 0], [1e-3, 1e-4, 0, 0]

        Args:
            lr (list): learning rate for different parameters [c, d, k, a]. Defaults to [1e-4, 1e-4, 0, 1e-4].
            decay (float): decay rate for higher order a. Defaults to 0.2.
            optim_surf_range (list): surface indices to be optimized. Defaults to None.
            optim_mat (bool): whether to optimize material. Defaults to False.

        Returns:
            list: optimizer parameters
        """
        if optim_surf_range is None:
            optim_surf_range = self.find_diff_surf()

        params = []
        for i in optim_surf_range:
            surf = self.surfaces[i]

            if isinstance(surf, Aperture):
                pass

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(
                    lr=lr, decay=decay, optim_mat=optim_mat
                )

            elif isinstance(surf, Diffractive_GEO):
                params += surf.get_optimizer_params(lr=lr[3])

            # elif isinstance(surf, GaussianRBF):
            #     params += surf.get_optimizer_params(lr=lr[3], optim_mat=optim_mat)

            # elif isinstance(surf, NURBS):
            #     params += surf.get_optimizer_params(lr=lr[3], optim_mat=optim_mat)

            elif isinstance(surf, Plane):
                pass

            # elif isinstance(surf, PolyEven):
            #     params += surf.get_optimizer_params(lr=lr, optim_mat=optim_mat)

            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(lr=lr[:2], optim_mat=optim_mat)

            elif isinstance(surf, Anamorphic):
                params += surf.get_optimizer_params(lr=lr[:2], optim_mat=optim_mat)

            else:
                raise Exception(
                    f"Surface type {surf.__class__.__name__} is not supported for optimization yet."
                )

        self.d_sensor.requires_grad = True
        params += [{"params": self.d_sensor, "lr": lr[1]}]

        return params

    def get_optimizer(self, lr=[1e-4, 1e-4, 0, 1e-4], decay=0.02, optim_mat=False):
        """Get optimizers and schedulers for different lens parameters.

        Args:
            lrs (_type_): _description_
            epochs (int, optional): _description_. Defaults to 100.
            ai_decay (float, optional): _description_. Defaults to 0.2.
        """
        params = self.get_optimizer_params(lr, decay, optim_mat=optim_mat)
        optimizer = torch.optim.Adam(params)
        return optimizer

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
        anamorphic=False,
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
        num_grid = 15
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
            loss_anamorphic_list = []
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
                
                if anamorphic:
                    # ---- Anamorphic (Squeeze) Loss Computation ----
                    # Horizontal offset ray.
                    ray_offset_h = self.get_cached_rays(depth=depth, wvln=wv, offset='h', num_grid=num_grid)
                    ray_offset_h = self.trace2sensor(ray_offset_h)
                    xy_offset_h = ray_offset_h.o[..., :2]
                    ra_xy_offset_h = ray_offset_h.ra.clone().detach()
                    xy_offset_h_norm = (xy_offset_h - center_p) * ra_xy_offset_h.unsqueeze(-1)

                    # Vertical offset ray.
                    ray_offset_v = self.get_cached_rays(depth=depth, wvln=wv, offset='v', num_grid=num_grid)
                    ray_offset_v = self.trace2sensor(ray_offset_v)
                    xy_offset_v = ray_offset_v.o[..., :2]
                    ra_xy_offset_v = ray_offset_v.ra.clone().detach()
                    xy_offset_v_norm = (xy_offset_v - center_p) * ra_xy_offset_v.unsqueeze(-1)
                    
                    delta = 1  # object-space offset (ensure consistent units)

                    # Compute effective magnifications.
                    M_x = torch.mean(torch.abs(xy_offset_h_norm[..., 0])) / delta
                    M_y = torch.mean(torch.abs(xy_offset_v_norm[..., 1])) / delta
                    print(f'Magnifications: {M_x}, {M_y}')
                    ratio = M_x / (M_y + EPSILON)
                    # Hardcoded for now
                    # TODO: Make this into a parameter
                    target_squeeze = 1.5
                    loss_anamorphic_list.append((ratio - target_squeeze) ** 2)

            loss_rms = sum(loss_rms) / len(loss_rms)
            loss_anamorphic = sum(loss_anamorphic_list) / len(loss_anamorphic_list)

            # Total loss
            loss_reg = self.loss_reg()
            w_reg = 0.1
            w_anamorphic = 0.5
            L_total = loss_rms + w_reg * loss_reg
            if anamorphic:
                L_total += w_anamorphic * loss_anamorphic

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=loss_rms.item())
            pbar.update(1)

        pbar.close()

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
                    s = Aspheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Cubic":
                    s = Cubic.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Diffractive_GEO":
                    s = Diffractive_GEO.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "GaussianRBF":
                #     s = GaussianRBF.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "NURBS":
                #     s = NURBS.init_from_dict(surf_dict)

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
                
                elif surf_dict["type"] == "Anamorphic":
                    s = Anamorphic.init_from_dict(surf_dict)

                else:
                    raise Exception(
                        f"Surface type {surf_dict['type']} is not implemented in GeoLens.read_lens_json()."
                    )

                self.surfaces.append(s)
                d += surf_dict["d_next"]

        self.d_sensor = torch.tensor(d)
        self.lens_info = data.get("info", "None")

        sensor_res = data.get("sensor_res", [1024, 1024])
        self.r_sensor = data["r_sensor"]
        self.set_sensor(sensor_res=sensor_res)

    def write_lens_json(self, filename="./test.json"):
        """Write the lens into .json file."""
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["foclen"] = round(self.foclen, 4)
        data["fnum"] = round(self.fnum, 4)
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

    def read_lens_zmx(self, filename="./test.zmx"):
        """Read the lens from .zmx file."""
        # Read ZMX file
        try:
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(filename, "r", encoding="utf-16") as file:
                lines = file.readlines()

        # Iterate through the lines and extract SURF dict
        surfs_dict = {}
        current_surf = None
        for line in lines:
            if line.startswith("SURF"):
                current_surf = int(line.split()[1])
                surfs_dict[current_surf] = {}
            elif current_surf is not None and line.strip() != "":
                if len(line.strip().split(maxsplit=1)) == 1:
                    continue
                else:
                    key, value = line.strip().split(maxsplit=1)
                    if key == "PARM":
                        new_key = "PARM" + value.split()[0]
                        new_value = value.split()[1]
                        surfs_dict[current_surf][new_key] = new_value
                    else:
                        surfs_dict[current_surf][key] = value

        # Print the extracted data for each SURF
        self.surfaces = []
        d = 0.0
        for surf_idx, surf_dict in surfs_dict.items():
            if surf_idx > 0 and surf_idx < current_surf:
                mat2 = (
                    f"{surf_dict['GLAS'].split()[3]}/{surf_dict['GLAS'].split()[4]}"
                    if "GLAS" in surf_dict
                    else "air"
                )
                surf_r = (
                    float(surf_dict["DIAM"].split()[0]) if "DIAM" in surf_dict else 1.0
                )
                surf_c = (
                    float(surf_dict["CURV"].split()[0]) if "CURV" in surf_dict else 0.0
                )
                surf_d_next = (
                    float(surf_dict["DISZ"].split()[0]) if "DISZ" in surf_dict else 0.0
                )

                if surf_dict["TYPE"] == "STANDARD":
                    # Aperture
                    if surf_c == 0.0 and mat2 == "air":
                        s = Aperture(r=surf_r, d=d)

                    # Spherical surface
                    else:
                        s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2)

                # Aspherical surface
                elif surf_dict["TYPE"] == "EVENASPH":
                    raise NotImplementedError()
                    s = Aspheric()

                else:
                    print(f"Surface type {surf_dict['TYPE']} not implemented.")
                    continue

                self.surfaces.append(s)
                d += surf_d_next

            elif surf_idx == current_surf:
                # Image sensor
                self.r_sensor = float(surf_dict["DIAM"].split()[0])

            else:
                pass

        self.d_sensor = torch.tensor(d)

    def write_lens_zmx(self, filename="./test.zmx"):
        """Write the lens into .zmx file."""
        lens_zmx_str = ""
        ENPD = self.calc_entrance_pupil()[1] * 2
        # Head string
        head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {ENPD}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707 * self.hfov * 57.3} {0.99 * self.hfov * 57.3}
WAVL 0.4861327 0.5875618 0.6562725
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
    TYPE STANDARD
    CURV 0.0
    DISZ INFINITY
"""
        lens_zmx_str += head_str

        # Surface string
        for i, s in enumerate(self.surfaces):
            d_next = (
                self.surfaces[i + 1].d - self.surfaces[i].d
                if i < len(self.surfaces) - 1
                else self.d_sensor - self.surfaces[i].d
            )
            surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
            lens_zmx_str += surf_str

        # Sensor string
        sensor_str = f"""SURF {i + 2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_sensor}
"""
        lens_zmx_str += sensor_str

        # Write lens zmx string into file
        with open(filename, "w") as f:
            f.writelines(lens_zmx_str)
            f.close()


# ====================================================================================
# Useful functions
# ====================================================================================
def create_lens(
    foclen,
    fov,
    fnum,
    flange,
    thickness=None,
    lens_type=[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]],
    save_dir="./",
):
    """Create a flat starting point for camera lens design.

    Contributor: Rayengineer

    Args:
        foclen: Focal length in mm.
        fov: Diagonal field of view in degrees.
        fnum: Maximum f number.
        flange: Distance from last surface to sensor.
        thickness: Total thickness if specified.
        lens_type: List of surface types defining each lens element and aperture.
    """
    # Compute lens parameters
    aper_r = foclen / fnum / 2
    imgh = 2 * foclen * float(np.tan(fov / 2 / 57.3))
    if thickness is None:
        thickness = foclen + flange
    d_opt = thickness - flange

    # Materials
    mat_names = list(SELLMEIER_TABLE.keys())
    remove_materials = ["air", "vacuum", "occluder"]
    for mat in remove_materials:
        if mat in mat_names:
            mat_names.remove(mat)

    # Create lens
    lens = GeoLens()
    surfaces = lens.surfaces

    d_total = 0.0
    for elem_type in lens_type:
        if elem_type == "Aperture":
            d_next = (torch.rand(1) + 0.5).item()
            surfaces.append(Aperture(r=aper_r, d=d_total))
            d_total += d_next

        elif isinstance(elem_type, list):
            if len(elem_type) == 1 and elem_type[0] == "Aperture":
                d_next = (torch.rand(1) + 0.5).item()
                surfaces.append(Aperture(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) in [2, 3]:
                for i, surface_type in enumerate(elem_type):
                    if i == len(elem_type) - 1:
                        mat = "air"
                        d_next = (torch.rand(1) + 0.5).item()
                    else:
                        mat = random.choice(mat_names)
                        d_next = (torch.rand(1) + 1.0).item()

                    surfaces.append(
                        create_surface(surface_type, d_total, aper_r, imgh, mat)
                    )
                    d_total += d_next
            else:
                raise Exception("Lens element type not supported yet.")
        else:
            raise Exception("Lens type format not correct.")

    # Normalize optical part total thickness
    d_opt_actual = d_total - d_next
    for s in surfaces:
        s.d = s.d / d_opt_actual * d_opt

    # Lens calculation
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness, dtype=torch.float32).to(lens.device)
    lens.r_sensor = imgh / 2
    lens.set_sensor(sensor_res=lens.sensor_res)
    lens.post_computation()

    # Save lens
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}.json"
    lens.write_lens_json(os.path.join(save_dir, filename))

    return lens


def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """Create a surface object based on the surface type."""
    if mat == "air":
         c = - np.random.rand() * 0.001
    else:
        c = np.random.rand() * 0.001
    r = max(imgh / 2, aper_r)

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)
    elif surface_type == "Aspheric":
        ai = np.random.randn(7).astype(np.float32) * 1e-30
        k = np.random.randn() * 0.001
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)
    elif surface_type == "Plane":
        return Plane(r=r, d=d_total, mat2=mat)
    elif surface_type == "Anamorphic":
        return Anamorphic(r=r, d=d_total, c_x=c, mat2=mat)
    else:
        raise Exception("Surface type not supported yet.")

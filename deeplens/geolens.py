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

from .lens import Lens
from .optics import DEPTH, EPSILON, PSF_KS, SELLMEIER_TABLE, SPP_CALC, SPP_PSF, Ray
from .optics.basics import (
    DEFAULT_WAVE,
    GEO_GRID,
    SPP_COHERENT,
    WAVE_RGB,
    init_device,
)
from .optics.materials import Material
from .optics.monte_carlo import forward_integral
from .optics.surfaces import (
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
            self.prepare_sensor()
            self.diff_surf_range = self.find_diff_surf()
            self.post_computation()

        else:
            self.sensor_res = [1024, 1024]
            self.surfaces = []
            self.materials = []
            self.to(self.device)

    def load_file(self, filename):
        """Load lens file."""
        if filename[-4:] == ".txt":
            raise Exception("File format support has been removed.")

        elif filename[-5:] == ".json":
            self.read_lens_json(filename)

        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)

        else:
            raise Exception("File format not supported.")

    def load_external(self, surfaces, materials, r_sensor, d_sensor):
        """Load lens from extrenal surface/material list.

        Args:
            surfaces (list): list of surfaces.
            materials (list): list of materials.
            r_sensor (float): sensor radius.
            d_sensor (float): sensor distance.
        """
        raise Warning("This function will be removed in the future.")
        self.surfaces = surfaces
        self.materials = materials
        self.r_sensor = r_sensor
        self.d_sensor = d_sensor

        for i in range(len(self.surfaces)):
            self.surfaces[i].mat1 = self.materials[i]
            self.surfaces[i].mat2 = self.materials[i + 1]

    def prepare_sensor(self, sensor_res=None, sensor_size=None):
        """Create sensor.

        Args:
            sensor_res (list): Resolution, pixel number.
            pixel_size (float): Pixel size in [mm].
        """
        if sensor_res is not None:
            sensor_res = (
                [sensor_res, sensor_res] if isinstance(sensor_res, int) else sensor_res
            )
            self.sensor_res = sensor_res

        H, W = self.sensor_res
        if sensor_size is None:
            self.sensor_size = [
                2 * self.r_sensor * H / math.sqrt(H**2 + W**2),
                2 * self.r_sensor * W / math.sqrt(H**2 + W**2),
            ]
        else:
            assert (
                sensor_size[0] * sensor_res[1] == sensor_res[0] * sensor_size[1]
            ), "Pixel is not square."
            self.sensor_size = sensor_size
            self.r_sensor = math.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2) / 2

        self.sensor_size = [float(self.sensor_size[0]), float(self.sensor_size[1])]
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]

    def post_computation(self):
        """After loading lens, compute foclen, fov and fnum."""
        self.find_aperture()
        self.hfov = self.calc_fov()
        self.foclen = self.calc_efl()

        avg_pupilz, avg_pupilx = self.entrance_pupil()
        self.fnum = self.foclen / avg_pupilx / 2

        # Constraints for lens design
        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.dist_min = 0.1
            self.dist_max = 0.6 #float("inf")
            self.thickness_min = 0.3
            self.thickness_max = 1.5
            self.flange_min = 0.5
            self.flange_max = float("inf")

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
            self.flange_max = float("inf")

            self.sag_max = 8.0
            self.grad_max = 1.0
            self.grad2_max = 100.0

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
        """Get surface indices without aperture."""
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )
        return diff_surf_range

    # ====================================================================================
    # Ray Sampling
    # ====================================================================================

    @torch.no_grad()
    def sample_parallel_2D(
        self,
        R=None,
        wvln=DEFAULT_WAVE,
        z=None,
        view=0.0,
        M=15,
        forward=True,
        entrance_pupil=False,
    ):
        """Sample 2D parallel rays. Rays have shape [M, 3].

            Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to infinity

        Args:
            R (float, optional): sampling radius. Defaults to None.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            z (float, optional): sampling depth. Defaults to None.
            view (float, optional): incident angle (in degree). Defaults to 0.0.
            M (int, optional): ray number. Defaults to 15.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
        """
        if entrance_pupil:
            # Sample 2nd points on the pupil
            pupilz, pupilx = self.entrance_pupil()

            x2 = torch.linspace(-pupilx, pupilx, M) * 0.99
            y2 = torch.zeros_like(x2)
            z2 = torch.full_like(x2, pupilz)
            o2 = torch.stack((x2, y2, z2), axis=-1)  # shape [M, 3]

            dx = torch.full_like(x2, np.sin(view / 57.3))
            dy = torch.zeros_like(x2)
            dz = torch.full_like(x2, np.cos(view / 57.3))
            d = torch.stack((dx, dy, dz), axis=-1)

            # Move ray origins to z = - 0.1 for tracing
            if pupilz > 0:
                o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
            else:
                o = o2

            return Ray(o, d, wvln, device=self.device)

        else:
            # Sample points on z = 0 or z = d_sensor
            x = torch.linspace(-R, R, M)
            y = torch.zeros_like(x)
            if z is None:
                z = 0 if forward else self.d_sensor.item()
            z = torch.full_like(x, z)
            o = torch.stack((x, y, z), axis=-1)

            # Calculate ray directions
            if forward:
                dx = torch.full_like(x, np.sin(view / 57.3))
                dy = torch.zeros_like(x)
                dz = torch.full_like(x, np.cos(view / 57.3))
            else:
                dx = torch.full_like(x, np.sin(view / 57.3))
                dy = torch.zeros_like(x)
                dz = torch.full_like(x, -np.cos(view / 57.3))

            d = torch.stack((dx, dy, dz), axis=-1)

            return Ray(o, d, wvln, device=self.device)

    @torch.no_grad()
    def sample_parallel(
        self,
        fov=0.0,
        R=None,
        z=None,
        M=15,
        wvln=DEFAULT_WAVE,
        sampling="grid",
        forward=True,
        entrance_pupil=False,
    ):
        """Sample parallel rays from plane (-R:R, -R:R, z). Rays have shape [spp, M, M, 3]

            Used for (1) in-focus loss, (2) RMS spot radius calculation (but not implemented)

        Args:
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            R (float, optional): sampling radius. Defaults to None.
            z (float, optional): sampling depth. Defaults to 0..
            M (int, optional): ray number. Defaults to 15.
            sampling (str, optional): sampling method. Defaults to 'grid'.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [spp, M, M]
        """
        if z is None:
            z = self.surfaces[0].d
        fov = np.radians(np.asarray(fov))  # convert degree to radian

        # Sample ray origins
        if entrance_pupil:
            pupilz, pupilr = self.entrance_pupil()
            if sampling == "grid":  # sample a square
                x, y = torch.meshgrid(
                    torch.linspace(-pupilr, pupilr, M),
                    torch.linspace(pupilr, -pupilr, M),
                    indexing="xy",
                )
            elif sampling == "radial":
                r2 = torch.rand((M, M)) * pupilr**2
                theta = torch.rand((M, M)) * 2 * torch.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception("Sampling method not implemented!")

        else:
            if R is None:
                # We want to sample at a depth, so radius of the cone need to be computed.
                sag = (
                    self.surfaces[0].surface(self.surfaces[0].r, 0.0).item()
                )  # sag is a float
                R = np.tan(fov) * sag + self.surfaces[0].r

            if sampling == "grid":  # sample a square
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M), torch.linspace(R, -R, M), indexing="xy"
                )
            elif sampling == "radial":
                r2 = torch.rand((M, M)) * R**2
                theta = torch.rand((M, M)) * 2 * torch.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception("Sampling method not implemented!")

        # Generate rays
        if isinstance(fov, float):
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2)
            d = torch.zeros_like(o)
            if forward:
                d[..., 2] = torch.full_like(x, np.cos(fov))
                d[..., 0] = torch.full_like(x, np.sin(fov))
            else:
                d[..., 2] = torch.full_like(x, -np.cos(fov))
                d[..., 0] = torch.full_like(x, -np.sin(fov))
        else:
            spp = len(fov)
            o = (
                torch.stack((x, y, torch.full_like(x, pupilz)), axis=2)
                .unsqueeze(0)
                .repeat(spp, 1, 1, 1)
            )
            d = torch.zeros_like(o)
            for i in range(spp):
                if forward:
                    d[i, :, :, 2] = torch.full_like(x, np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, np.sin(fov[i]))
                else:
                    d[i, :, :, 2] = torch.full_like(x, -np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, -np.sin(fov[i]))

        rays = Ray(o, d, wvln, device=self.device)
        rays.propagate_to(z)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self, depth=-1000, view=0, M=7, entrance_pupil=False, wvln=DEFAULT_WAVE
    ):
        """Sample point source 2D rays. Rays hape shape of [M, 3].

            Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to given depth

        Args:
            depth (float, optional): sampling depth. Defaults to -1000.
            view (float, optional): incident angle (in degree). Defaults to 0.
            M (int, optional): ray number. Defaults to 9.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [M, 3]
        """
        if entrance_pupil:
            pupilz, pupilx = self.entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        # Second point on the pupil or first surface
        x2 = torch.linspace(-pupilx, pupilx, M, device=self.device) * 0.98
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=1)

        # First point is the point source
        o1 = torch.zeros_like(o2)
        o1[:, 2] = depth
        o1[:, 0] = depth * math.tan(view / 57.3)

        # Form the rays and propagate to z = 0
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=self.device)

        return ray

    @torch.no_grad()
    def sample_point_source(
        self,
        R=None,
        depth=-10.0,
        M=11,
        spp=SPP_PSF,
        fov=10.0,
        forward=True,
        pupil=True,
        wvln=DEFAULT_WAVE,
        importance_sampling=False,
    ):
        """Sample forward point-grid rays. Rays have shape [spp, M, M, 3]

            Rays come from a 2D square array (-R~R, -Rw~Rw, depth), and fall into a cone spercified by fov or pupil.

            Equivalent to self.point_source_grid() + self.sample_from_points()

            Used for (1) spot/rms/magnification calculation, (2) distortion/sensor sampling

        Args:
            R (float, optional): sample plane half side length. Defaults to None.
            depth (float, optional): sample plane z position. Defaults to -10.0.
            spp (int, optional): sample per pixel. Defaults to 16.
            fov (float, optional): cone angle. Defaults to 10.0.
            M (int, optional): sample plane resolution. Defaults to 11.
            forward (bool, optional): forward or backward rays. Defaults to True.
            pupil (bool, optional): whether to use pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [spp, M, M, 3]
        """
        if R is None:
            R = self.surfaces[0].r
        Rw = R * self.sensor_res[1] / self.sensor_res[0]  # half height

        # sample o
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, M), torch.linspace(1, -1, M), indexing="xy"
        )
        if importance_sampling:
            x = torch.sqrt(x.abs()) * x.sign()
            y = torch.sqrt(y.abs()) * y.sign()

        x = x * Rw
        y = y * R

        z = torch.full_like(x, depth)
        o = torch.stack((x, y, z), -1).to(self.device)
        o = o.unsqueeze(0).repeat(spp, 1, 1, 1)

        # sample d
        if pupil:
            if depth < 0:
                pupilz, pupilr = self.entrance_pupil()
            else:
                pupilz, pupilr = self.exit_pupil()
            o2 = self.sample_pupil(res=(M, M), spp=spp, pupilr=pupilr, pupilz=pupilz)
            d = o2 - o
            d = d / torch.linalg.vector_norm(d, ord=2, dim=-1, keepdim=True)
        else:
            raise Exception(
                "Cone sampling specified by fov has been abandoned. Use pupil sampling instead."
            )

        ray = Ray(o, d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        o=[[0.0, 0.0, -10000.0]],
        spp=256,
        wvln=DEFAULT_WAVE,
        shrink_pupil=False,
        normalized=False,
    ):
        """Sample forward rays from given point source (un-normalized positions). Rays have shape [spp, N, 3]

            Used for (1) PSF calculation, (2) chief ray calculation.

        Args:
            o (list): ray origin. Defaults to [[0, 0, -10000]].
            spp (int): sample per pixel. Defaults to 8.
            forward (bool): forward or backward rays. Defaults to True.
            pupil (bool): whether to use pupil. Defaults to True.
            fov (float): cone angle. Defaults to 10.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray: Ray object. Shape [spp, N, 3]
        """
        if normalized:
            raise NotImplementedError("This feature has not been implemented yet.")
        else:
            o = torch.tensor(o) if not torch.is_tensor(o) else o

        # Sample pupil and compute d
        pupilz, pupilr = self.entrance_pupil(shrink_pupil=shrink_pupil)
        theta = torch.rand(spp) * 2 * torch.pi
        r = torch.sqrt(torch.rand(spp) * pupilr**2 + EPSILON)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), 1)

        if len(o.shape) == 2:
            o = o.unsqueeze(0).repeat(spp, 1, 1)  # shape [spp, N, 3]
            d = o2.unsqueeze(1).to(o.device) - o
        elif len(o.shape) == 3:
            o = o.unsqueeze(0).repeat(spp, 1, 1, 1)
            d = o2.unsqueeze(1).unsqueeze(1).to(o.device) - o
        else:
            raise Exception("Input o has wrong shape.")

        # Calculate rays
        ray = Ray(o, d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_sensor(self, spp=64, pupil=True, wvln=DEFAULT_WAVE, sub_pixel=False):
        """Sample rays from sensor pixels. Rays have shape of [spp, H, W, 3].

        Args:
            sensor_scale (int, optional): number of pixels remain the same, but only sample rays on part of sensor. Defaults to 1.
            spp (int, optional): sample per pixel. Defaults to 1.
            vpp (int, optional): sample per pixel on pupil. Defaults to 64.
            high_spp (bool, optional): whether to use high spp. Defaults to False.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [spp, H, W, 3]
        """
        # ===> sample o1 on sensor plane
        # We use top-left point as reference in rendering, so here we should sample bottom-right point
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
        z1 = torch.full_like(x1, self.d_sensor.item(), device=self.device)

        # ==> Sample o2 on the second plane and compute rays
        if pupil is True:
            pupilz, pupilr = self.exit_pupil()
        else:
            raise Exception("This feature has been abandoned.")
            pupilz, pupilr = self.surfaces[-1].d.item(), self.surfaces[-1].r

        if sub_pixel:
            # For more realistic rendering, we can sample multiple points inside the pixel
            raise Warning("This feature is not finished yet.")

        else:
            # Use bottom-right corner to represent each pixel
            # sample o2, method 2, o2 shape [spp, res, res, 3]
            o2 = self.sample_pupil(self.sensor_res, spp, pupilr=pupilr, pupilz=pupilz)

            o = torch.stack((x1, y1, z1), 2)
            o = torch.broadcast_to(o, o2.shape)
            d = o2 - o  # broadcast to [spp, H, W, 3]

        ray = Ray(o, d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_pupil(
        self, res=(512, 512), spp=16, num_angle=8, pupilr=None, pupilz=None
    ):
        """Sample points (not rays) on the pupil plane with rings. Points have shape [spp, res, res].

            2*pi is devided into [num_angle] sectors.
            Circle is devided into [spp//num_angle] rings.

        Args:
            res (tuple): pupil plane resolution. Defaults to (512,512).
            spp (int): sample per pixel. Defaults to 16.
            num_angle (int): number of sectors. Defaults to 8.
            pupilr (float): pupil radius. Defaults to None.
            pupilz (float): pupil z position. Defaults to None.
            multiplexing (bool): whether to use multiplexing. Defaults to False.

        Returns:
            o (torch.Tensor): Ray origins. Shape [spp, res, res, 3]
        """
        H, W = res
        if pupilr is None or pupilz is None:
            pupilz, pupilr = self.entrance_pupil()

        # => Naive implementation
        if spp % num_angle != 0 or spp >= 10000:
            theta = torch.rand((spp, H, W), device=self.device) * 2 * torch.pi
            r2 = torch.rand((spp, H, W), device=self.device) * pupilr**2
            r = torch.sqrt(r2 + EPSILON)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x, y, z), -1)

        # => Sample more uniformly when spp is not large
        else:
            # ==> For each pixel, sample different points on the pupil
            x, y = [], []
            for i in range(num_angle):
                for j in range(spp // num_angle):
                    # sample delta_theta from [0, pi/4)
                    delta_theta = (
                        torch.rand((1, *res), device=self.device)
                        * 2
                        * torch.pi
                        / num_angle
                    )
                    theta = delta_theta + i * 2 * torch.pi / num_angle

                    delta_r2 = (
                        torch.rand((1, *res), device=self.device)
                        * pupilr**2
                        / spp
                        * num_angle
                    )
                    r2 = delta_r2 + j * pupilr**2 / spp * num_angle
                    r = torch.sqrt(r2 + EPSILON)

                    x.append(r * torch.cos(theta))
                    y.append(r * torch.sin(theta))

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x, y, z), -1)

        return o

    # ====================================================================================
    # Ray Tracing functions
    # ====================================================================================

    def trace(self, ray, lens_range=None, record=False):
        """General ray tracing function. Ray in and ray out.

            Transform between local and world coordinates and do ray tracing under local coordinates.

            Forward or backward ray tracing is automatically determined by ray directions.

        Args:
            ray ([type]): [description]
            stop_ind ([int]): Early stop index.
            record: Only when we want to plot ray path, set `record` to True.

        Returns:
            ray_final (Ray object): ray after optical system.
            valid (boolean matrix): mask denoting valid rays.
            oss (list): list of intersection points.
        """
        is_forward = ray.d.reshape(-1, 3)[0, 2] > 0
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))

        if is_forward:
            valid, ray_out, oss = self.forward_tracing(ray, lens_range, record=record)
        else:
            valid, ray_out, oss = self.backward_tracing(ray, lens_range, record=record)

        return ray_out, valid, oss

    def trace2obj(self, ray, depth=DEPTH):
        """Trace rays through the lens and reach the sensor plane.

        Args:
            ray (Ray object): Ray object.
            depth (float): sensor distance.
        """
        (
            ray,
            _,
            _,
        ) = self.trace(ray)
        ray = ray.propagate_to(depth)
        return ray

    def trace2sensor(self, ray, record=False, ignore_invalid=False):
        """Trace optical rays to sensor plane.

        Args:
            ray (Ray object): Ray object.
            record (bool): record ray path or not.
            ignore_invalid (bool): ignore invalid rays or not.

        Returns:
            p (torch.Tensor): intersection points.
            oss (list): list of intersection points.
        """
        if record:
            ray_out, valid, oss = self.trace(ray, record=record)
            ray_out = ray_out.propagate_to(self.d_sensor)
            valid = ray_out.ra == 1
            p = ray.o
            for os, v, pp in zip(
                oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()
            ):
                if v.any():
                    os.append(pp)

            if ignore_invalid:
                p = p[valid]
            else:
                assert len(p.shape) >= 2, "This function is not tested."
                p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))

            for v, os, pp in zip(valid, oss, p):
                if v:
                    os.append(pp.cpu().detach().numpy())
            return p, oss

        else:
            (
                ray,
                _,
                _,
            ) = self.trace(ray)
            ray = ray.propagate_to(self.d_sensor)
            return ray

    def forward_tracing(self, ray, lens_range, record):
        """Trace rays from object space to sensor plane.

        Args:
            ray (Ray object): Ray object.
            lens_range (list): lens range.
            record (bool): record ray path or not.
        """
        dim = ray.o[
            ..., 2
        ].shape  # What does this mean: how many rays do we have? here 31*31

        if record:
            oss = []  # oss records all points of intersection. ray.o shape of [N, 3]
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])
        else:
            oss = None

        mat1 = Material("air")
        for i in lens_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            valid = ray.ra == 1
            if record:
                p = ray.o
                for os, v, pp in zip(
                    oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()
                ):
                    if v.any():
                        os.append(pp)

        valid = ray.ra == 1
        return valid, ray, oss

    def backward_tracing(self, ray, lens_range, record):
        """Trace rays from sensor plane to object space.

        Args:
            ray (Ray object): Ray object.
            lens_range (list): lens range.
            record (bool): record ray path or not.
        """
        dim = ray.o[..., 2].shape
        valid = ray.ra == 1

        if record:
            oss = []  # oss records all points of intersection
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])
        else:
            oss = None

        mat1 = Material("air")
        for i in np.flip(lens_range):
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i - 1].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i - 1].mat2

            valid = ray.ra > 0
            if record:
                p = ray.o
                for os, v, pp in zip(
                    oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()
                ):
                    if v.any():
                        os.append(pp)

        valid = ray.ra == 1
        return valid, ray, oss

    # ====================================================================================
    # Ray-tracing based rendering
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
                img_render, field_channel = self.render_psf_patch(img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks)    
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

    def render_raytracing(self, img, depth=DEPTH, spp=64, vignetting=False):
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
        scale = self.calc_scale_pinhole(depth=depth)
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
            ray (Ray object): Ray object. Shape [spp, H, W].
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
        # (img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], irr_img shape [N, C, spp, H, W])
        irr_img = img[..., idx_i, idx_j] * w_i * w_j
        irr_img += img[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
        irr_img += img[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
        irr_img += img[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)

        # Computation image
        if not vignetting:
            image = torch.sum(irr_img * ray.ra, -3) / (torch.sum(ray.ra, -3) + EPSILON)
        else:
            image = torch.sum(irr_img * ray.ra, -3) / torch.numel(ray.ra)

        return image

    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        depth=DEPTH,
        spp=64,
        unwarp=False,
        save_name=None,
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

    # ====================================================================================
    # PSF and spot diagram (incoherent ray tracing)
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
            # Shrink the pupil and calculate centroid ray as the chief ray. Distortion is allowed.
            ray = self.sample_from_points(point, spp=SPP_CALC, shrink_pupil=True)
            ray = self.trace2sensor(ray)
            assert (ray.ra == 1).any(), "No sampled rays is valid."
            psf_center = (ray.o * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(
                -1
            ).sum(0).add(
                EPSILON
            )  # shape [N, 3]
            psf_center = -psf_center[..., :2]  # shape [N, 2]

        elif method == "pinhole":
            # Pinhole camera perspective projection. This doesnot allow distortion.
            scale = self.calc_scale_pinhole(point[..., 2])
            psf_center = point[..., :2] / scale.unsqueeze(-1)

        else:
            raise Exception("Unsupported method.")

        return psf_center

    def psf(self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_PSF, center=True):
        """Single wvln incoherent PSF calculation.

        Args:
            points (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 51.
            spp (int, optional): Sample per pixel. For diff ray tracing, usually kernel_size^2. Defaults to 2048.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            kernel: Shape of [N, ks, ks] or [ks, ks].
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
        point_obj_x = points[..., 0] * scale * self.sensor_size[1] / 2  # x coordinate
        point_obj_y = points[..., 1] * scale * self.sensor_size[0] / 2  # y coordinate
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)

        # Trace rays to sensor plane
        ray = self.sample_from_points(o=point_obj, spp=spp, wvln=wvln)
        ray = self.trace2sensor(ray)

        # Calculate PSF
        if center:
            # PSF center on the sensor plane defined by chief ray
            pointc_chief_ray = self.psf_center(point_obj)  # shape [N, 2]
            psf = forward_integral(
                ray,
                ps=self.pixel_size,
                ks=ks,
                pointc_ref=pointc_chief_ray,
                coherent=False,
            )
        else:
            # PSF center on the sensor plane defined by pespective
            pointc_ideal = points.clone()[:, :2]
            pointc_ideal[:, 0] *= self.sensor_size[1] / 2
            pointc_ideal[:, 1] *= self.sensor_size[0] / 2
            psf = forward_integral(
                ray, ps=self.pixel_size, ks=ks, pointc_ref=pointc_ideal, coherent=False
            )

        # Normalize to 1
        psf = psf / (psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 10e-9)

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
        center=True,
    ):
        """Computes the PSF map at a specified depth. This implementation overrides the base method to improve efficiency through parallel computation.

        Args:
            depth (float, optional): Depth of the point source plane. Defaults to DEPTH.
            grid (int, optional): Grid size. Defaults to 7.
            ks (int, optional): Kernel size. Defaults to 51.
            spp (int, optional): Sample per pixel. Defaults to None.
            center (bool, optional): Use spot center as PSF center. Defaults to True.

        Returns:
            psf_map: Shape of [grid*ks, grid*ks].
        """
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, center=center, spp=spp, wvln=wvln
        ).unsqueeze(
            1
        )  # shape [grid**2, 1, ks, ks]

        psf_map = make_grid(psfs, nrow=grid, padding=0)[
            0, :, :
        ]  # shape [grid*ks, grid*ks]
        return psf_map

    @torch.no_grad()
    def rms_map(self, res=(128, 128), depth=DEPTH):
        """Calculate the RMS spot error map as a weight mask for lens design.

        Args:
            res (tuple, optional): resolution of the RMS map. Defaults to (32, 32).
            depth (float, optional): depth of the point source. Defaults to DEPTH.

        Returns:
            rms_map (torch.Tensor): RMS map normalized to [0, 1].
        """
        M = 128
        scale = -depth * np.tan(self.hfov) / self.r_sensor
        ray = self.sample_point_source(
            M=M, depth=depth, R=self.sensor_size[0] / 2 * scale, pupil=True
        )
        ray, _, _ = self.trace(ray)
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

    def analysis_rms(self, depth=DEPTH):
        """Compute RMS-based error. Contain both RMS errors and RMS radius."""
        grid = 20
        x = torch.linspace(0, 1, grid)
        y = torch.linspace(0, 1, grid)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)
        scale = self.calc_scale_ray(depth)

        # Ray position in the object space by perspective projection, because points are normalized
        point_obj_x = points[..., 0] * scale * self.sensor_size[1] / 2  # x coordinate
        point_obj_y = points[..., 1] * scale * self.sensor_size[0] / 2  # y coordinate
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)

        # Point center determined by green light
        ray = self.sample_from_points(o=point_obj, spp=SPP_PSF, wvln=DEFAULT_WAVE)
        ray = self.trace2sensor(ray)
        pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(
            0
        ).add(EPSILON).unsqueeze(-1)

        # Calculate RMS spot size
        rms = []
        for wvln in WAVE_RGB:
            # Trace rays to sensor plane
            ray = self.sample_from_points(o=point_obj, spp=SPP_PSF, wvln=wvln)
            ray = self.trace2sensor(ray)

            # Calculate RMS error for different FoVs
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.ra.unsqueeze(-1)
            rms0 = torch.sqrt(
                (o2_norm**2 * ray.ra.unsqueeze(-1)).sum((0, 2))
                / (ray.ra.sum(0) + EPSILON)
            )
            rms.append(rms0)

        rms = torch.stack(rms, dim=0)
        rms = torch.mean(rms, dim=0)

        # Calculate RMS error for on-axis and off-axis
        rms_avg = rms.mean()
        rms_radius_on_axis = rms[0]
        rms_radius_off_axis = rms[-1]

        return rms_avg, rms_radius_on_axis, rms_radius_off_axis

    # ====================================================================================
    # Coherent ray tracing
    # ====================================================================================

    def pupil_field(self, point, wvln=DEFAULT_WAVE, spp=SPP_COHERENT):
        """Compute complex wavefront (flipped for further PSF calculation) at exit pupil plane by coherent ray tracing. The wavefront has the same size as image sensor.

            This function is differentiable.

        Args:
            point (tensor): Point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            wvln (float): Ray wavelength in [um].
            spp (int): Ray sample number per point.
        """
        assert (
            spp >= 1000000
        ), "Coherent ray tracing spp is too small, will cause inaccurate simulation."
        assert (
            torch.get_default_dtype() == torch.float64
        ), "Please set the default dtype to float64 for accurate phase calculation."

        if len(point.shape) == 1:
            point = point.unsqueeze(0)  # shape of [1, 3]

        # Ray origin in the object space
        scale = self.calc_scale_ray(point[:, 2].item())
        point_obj = point.clone()
        point_obj[:, 0] = point[:, 0] * scale * self.sensor_size[1] / 2  # x coordinate
        point_obj[:, 1] = point[:, 1] * scale * self.sensor_size[0] / 2  # y coordinate

        # Ray center determined by chief ray
        # shape [N, 2], un-normalized physical coordinates
        pointc_chief_ray = self.psf_center(point_obj)

        # Ray-tracing to last surface
        ray = self.sample_from_points(o=point_obj, spp=spp, wvln=wvln)
        ray.coherent = True
        ray, _, _ = self.trace(ray)

        # Back-trace to exit pupil plane
        pupilz, pupilr = self.exit_pupil()
        ray = ray.prop_to(pupilz)

        # Calculate a full-resolution complex field for exit-pupil diffraction
        pointc_ref = torch.zeros_like(point[:, :2])
        wavefront = forward_integral(
            ray,
            ps=self.pixel_size,
            ks=self.sensor_res[0],
            pointc_ref=pointc_ref,
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
        pupilz, pupilr = self.exit_pupil()
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

    # ====================================================================================
    # Geometrical optics
    #   1. Focus-related functions
    #   2. FoV-related functions
    #   3. Pupil-related functions
    # ====================================================================================

    # ---------------------------
    # 1. Focus-related functions
    # ---------------------------
    def calc_foclen(self):
        """Calculate the focus length."""
        if (
            self.r_sensor < 8
        ):  # Cellphone lens, we usually use EFL to describe the lens.
            return self.calc_efl()
        else:  # Camera lens, we use the to describe the lens.
            return self.calc_bfl()

    def calc_bfl(self, wvln=DEFAULT_WAVE):
        """Compute back focal length (BFL).

        BFL: Distance from the second principal point to focal plane.
        """
        M = GEO_GRID

        # Forward ray tracing
        ray = self.sample_parallel_2D(
            R=self.surfaces[0].r * 0.5, M=M, forward=True, wvln=wvln
        )
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

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
    def calc_foc_dist(self, wvln=DEFAULT_WAVE):
        """Compute the focus distance (object space) of the lens. Ray starts from sensor and trace to the object space.

        Returns:
            focus_dist: Focus distance in object space. Negative value.
        """
        # Sample point source rays from sensor center
        o1 = torch.tensor([0, 0, self.d_sensor.item()], device=self.device).repeat(
            SPP_CALC, 1
        )

        # Sample the first surface as pupil
        o2 = self.surfaces[0].surface_sample(SPP_CALC)
        o2 *= 0.25  # Shrink sample region to improve accuracy
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=self.device)

        # Trace rays to object space
        ray, _, _ = self.trace(ray)

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
    def foc_plane(self):
        """Calculate in-focus sensor plane."""
        # Trace rays and compute in-focus sensor position
        ray = self.sample_parallel_2D(
            R=self.surfaces[0].r * 0.5, M=SPP_CALC, wvln=DEFAULT_WAVE
        )
        ray, _, _ = self.trace(ray)
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_p = ray.o[..., 2] - ray.d[..., 2] * t
        focus_p = focus_p[ray.ra > 0]
        focus_p = focus_p[~torch.isnan(focus_p) & (focus_p > 0)]
        infocus_sensor_d = torch.mean(focus_p)

        return infocus_sensor_d

    @torch.no_grad()
    def refocus_inf(self):
        """Move sensor to get the infinite focusing."""
        d_sensor_new = self.foc_plane()

        # Update sensor position
        assert d_sensor_new > 0, "sensor position is negative."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def refocus(self, depth=DEPTH):
        """Refocus the lens to a depth distance by changing sensor position.

        In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Trace green light
        o = self.surfaces[0].surface_sample(SPP_CALC)
        d = o - torch.tensor([0, 0, depth]).to(self.device)
        ray = Ray(o, d, device=self.device)
        ray, _, _ = self.trace(ray)

        # Calculate in-focus sensor position of green light (use least-squares solution)
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        t = t * ray.ra
        focus_d = ray.o[..., 2] - ray.d[..., 2] * t
        focus_d = focus_d[ray.ra > 0]
        focus_d = focus_d[~torch.isnan(focus_d) & (focus_d > 0)]
        d_sensor_new = torch.mean(focus_d)

        # Update sensor position
        assert d_sensor_new > 0, "sensor position is negative."
        self.d_sensor = d_sensor_new  # d_sensor should be a float value, not a tensor

        # FoV will be slightly changed
        self.post_computation()

    # ---------------------------
    # 2. FoV-related functions
    # ---------------------------

    @torch.no_grad()
    def calc_fov(self):
        """Compute half diagonal fov.

        Shot rays from edge of sensor, trace them to the object space and compute
        angel, output rays should be parallel and the angle is half of fov.
        """
        # Sample rays going out from edge of sensor, shape [M, 3]
        o1 = torch.zeros([SPP_CALC, 3])
        o1 = torch.tensor([self.r_sensor, 0, self.d_sensor.item()]).repeat(SPP_CALC, 1)

        # Option 1: sample second points on exit pupil
        # pupilz, pupilx = self.exit_pupil()
        # x2 = torch.linspace(-pupilx, pupilx, SPP_CALC)
        # z2 = torch.full_like(x2, pupilz)
        # Option 2: sample second points on last surface
        x2 = torch.linspace(0, self.surfaces[-1].r, SPP_CALC)
        z2 = torch.full_like(x2, self.surfaces[-1].d.item())

        y2 = torch.full_like(x2, 0)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        ray = self.trace2obj(ray)

        # compute fov
        tan_fov = ray.d[..., 0] / ray.d[..., 2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))

        if torch.isnan(fov):
            print("computed fov is NaN, use 0.5 rad instead.")
            fov = 0.5
        else:
            fov = fov.item()

        return fov

    @torch.no_grad()
    def calc_magnification3(self, depth):
        """Use mapping relationship (ray tracing) to compute magnification. The computed magnification is very accurate.

        Advatages: can use many data points to reduce error.
        Disadvantages: due to distortion, some data points contain error
        """
        M = GEO_GRID
        spp = 512

        # Sample rays [spp, W, H] from the object plane
        ray = self.sample_point_source(
            M=M, spp=spp, depth=depth, R=-depth * np.tan(self.hfov) * 0.7, pupil=True
        )

        # Map r1 from object space to sensor space, ground-truth
        o1 = ray.o.detach()[..., :2]
        o1 = torch.flip(o1, [1, 2])

        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)

        # Use 1/4 part of regions to compute magnification, also to avoid zero values on the axis
        x1 = o1[0, :, :, 0]
        y1 = o1[0, :, :, 1]
        x2 = torch.sum(o2[..., 0] * ray.ra, axis=0) / torch.sum(ray.ra, axis=0).add(
            EPSILON
        )
        y2 = torch.sum(o2[..., 1] * ray.ra, axis=0) / torch.sum(ray.ra, axis=0).add(
            EPSILON
        )

        mag_x = x1 / x2
        tmp = mag_x[: M // 2, : M // 2]
        mag = 1 / torch.mean(tmp[~tmp.isnan()]).item()

        if mag == 0:
            scale = -depth * np.tan(self.hfov) / self.r_sensor
            return 1 / scale

        return mag

    @torch.no_grad()
    def calc_principal(self, wvln=DEFAULT_WAVE):
        """Compute principal (front and back) planes."""
        M = GEO_GRID

        # Backward ray tracing for the first principal point
        ray = self.sample_parallel_2D(
            R=self.surfaces[0].r * 0.1, M=M, forward=False, wvln=wvln
        )
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        front_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        # Forward ray tracing for the second principal point
        ray = self.sample_parallel_2D(
            R=self.surfaces[0].r * 0.1, M=M, forward=True, wvln=wvln
        )
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        back_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        return front_principal, back_principal

    @torch.no_grad()
    def calc_scale(self, depth, method="pinhole"):
        """Calculate the scale factor."""
        if method == "pinhole":
            scale = self.calc_scale_pinhole(depth)
        elif method == "ray":
            scale = self.calc_scale_ray(depth)
        else:
            raise ValueError("Invalid method.")

        return scale

    @torch.no_grad()
    def calc_scale_pinhole(self, depth):
        """Assume the first principle point is at (0, 0, 0), use pinhole camera to calculate the scale factor."""
        scale = -depth * np.tan(self.hfov) / self.r_sensor
        return scale

    @torch.no_grad()
    def calc_scale_ray(self, depth):
        """Use ray tracing to compute scale factor."""
        if isinstance(depth, torch.Tensor) and len(depth.shape) == 1:
            scale = []
            for d in depth:
                scale.append(1 / self.calc_magnification3(d))
            scale = torch.tensor(scale)
        else:
            scale = 1 / self.calc_magnification3(depth)

        return scale

    @torch.no_grad()
    def chief_ray(self):
        """Compute chief ray. We can use chief ray for fov, magnification.
        Chief ray, a ray goes through center of aperture.
        """
        # sample rays with shape [M, 3]
        M = 1000
        pupilz, pupilx = self.exit_pupil()
        o1 = torch.zeros([M, 3])
        o1[:, 0] = pupilx
        o1[:, 2] = self.d_sensor.item()

        x2 = torch.linspace(-pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        inc_ray = ray.clone()
        ray, _, _ = self.trace(
            ray, lens_range=list(range(self.aper_idx, len(self.surfaces)))
        )

        center_x = torch.min(torch.abs(ray.o[:, 0]))
        center_idx = torch.where(torch.abs(ray.o[:, 0]) == center_x)

        return inc_ray.o[center_idx, :], inc_ray.d[center_idx, :]

    # ---------------------------
    # 3. Pupil-related functions
    # ---------------------------
    @torch.no_grad()
    def exit_pupil(self, shrink_pupil=False):
        """Sample **forward** rays to compute z coordinate and radius of exit pupil. Exit pupil: ray comes from sensor to object space.

        Reference: https://en.wikipedia.org/wiki/Exit_pupil
        """
        return self.entrance_pupil(entrance=False, shrink_pupil=shrink_pupil)

    @torch.no_grad()
    def entrance_pupil(self, M=SPP_CALC, entrance=True, shrink_pupil=False):
        """Sample **backward** rays, return z coordinate and radius of entrance pupil. Entrance pupil: how many rays can come from object space to sensor.

        Reference: https://en.wikipedia.org/wiki/Entrance_pupil "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
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
        ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(M, 1)

        # Sample phi ranges from [-0.5rad, 0.5rad]
        phi = torch.linspace(-0.5, 0.5, M)
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
            lens_range = range(0, self.aper_idx)
            ray, _, _ = self.trace(ray, lens_range=lens_range)
        else:
            lens_range = range(self.aper_idx + 1, len(self.surfaces))
            ray, _, _ = self.trace(ray, lens_range=lens_range)

        # Compute intersection points. o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.ra != 0][:, 0], ray.o[ray.ra != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.ra != 0][:, 0], ray.d[ray.ra != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)
        if len(intersection_points) == 0:
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
    #   1. Set lens parameters
    #   2. Lens operation (init, reverse, spherize), will be abandoned
    #   3. Lens pruning
    # ====================================================================================

    # ---------------------------
    # 1. Set lens parameters
    # ---------------------------

    def set_aperture(self, fnum=None, foclen=None, aper_r=None):
        """Change aperture radius."""
        if aper_r is None:
            if foclen is None:
                foclen = self.calc_efl()
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r

        self.fnum = self.foclen / aper_r / 2

    def set_target_fov_fnum(self, hfov, fnum, imgh=None):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        This function now only works for aperture in the front
        """
        if imgh is not None:
            self.r_sensor = imgh / 2
        self.hfov = hfov
        self.fnum = fnum

        self.foclen = self.calc_efl()
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].r = float(aper_r)

    # ---------------------------
    # 2. Lens operation
    # ---------------------------

    def pertub(self):
        """Randomly perturb all lens surfaces to simulate manufacturing errors.

        Including:
            (1) surface position, thickness, curvature, and other coefficients.
            (2) surface rotation, tilt, and decenter.

        Called for accurate image simulation, together with sensor noise, vignetting, etc.
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].perturb()

    def double(self):
        """Use double-precision for the lens group."""
        torch.set_default_dtype(torch.float64)
        for surf in self.surfaces:
            surf.double()

    # ---------------------------
    # 3. Lens pruning
    # ---------------------------

    @torch.no_grad()
    def prune_surf(self, expand_surf=None, surface_range=None):
        """Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            expand_surf (float): extra height to reserve.
                For cellphone lens, we usually use 0.1mm or 0.05 * r_sensor.
                For camera lens, we usually use 0.5mm or 0.1 * r_sensor.
        """
        surface_range = (
            self.find_diff_surf() if surface_range is None else surface_range
        )

        if self.is_cellphone:
            expand_surf = 0.05 if expand_surf is None else expand_surf

            # ==> 1. Reset lens to maximum height(sensor radius)
            for i in surface_range:
                # self.surfaces[i].r = self.r_sensor
                self.surfaces[i].r = max(self.r_sensor, self.surfaces[self.aper_idx].r)

            # ==> 2. Prune to reserve valid surface height
            # sample maximum fov rays to compute valid surface height
            view = (
                self.hfov
                if self.hfov is not None
                else np.arctan(self.r_sensor / self.d_sensor.item())
            )
            ray = self.sample_parallel_2D(
                view=np.rad2deg(view), M=GEO_GRID, entrance_pupil=True
            )
            # ray = self.sample_parallel_2D(
            #     view=np.rad2deg(view), M=SPP_CALC, entrance_pupil=True
            # )

            ps, oss = self.trace2sensor(ray=ray, record=True)
            for i in surface_range:
                height = []
                for os in oss:  # iterate all rays
                    try:
                        # because oss records the starting point at position 0, we need to ignore this.
                        # the second index 0 means x coordinate
                        height.append(os[i + 1][0].abs().item())
                    except:
                        continue

                try:
                    self.surfaces[i].r = max(height) * (1 + expand_surf)
                except:
                    continue

            # ==> 4. Remove nan part, also the maximum height should not exceed sensor radius
            for i in surface_range:
                # max_height = min(self.surfaces[i].max_height(), self.r_sensor)
                max_height = self.surfaces[i].max_height()
                self.surfaces[i].r = min(self.surfaces[i].r, max_height)

        else:
            expand_surf = 0.5 if expand_surf is None else expand_surf

            # sample maximum fov rays to compute valid surface height
            view = (
                self.hfov
                if self.hfov is not None
                else np.arctan(self.r_sensor / self.d_sensor.item())
            )
            ray = self.sample_parallel_2D(
                view=np.rad2deg(view), M=21, entrance_pupil=True
            )

            ps, oss = self.trace2sensor(ray=ray, record=True)
            for i in surface_range:
                height = []
                for os in oss:  # iterate all rays
                    try:
                        # Because oss records the starting point at position 0, we need to ignore this.
                        # The second index 0 means x coordinate
                        height.append(os[i + 1][0].abs().item())
                    except:
                        continue

                try:
                    self.surfaces[i].r = max(height) * (1 + expand_surf)
                except:
                    continue

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match material"""
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)

    @torch.no_grad()
    def correct_shape(self, expand_surf=None):
        """Correct wrong lens shape during the lens design."""
        aper_idx = self.aper_idx
        diff_surf_range = self.find_diff_surf()
        shape_changed = False

        # ==> Rule 1: Move the first surface to z = 0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # ==> Rule 2: Move lens group to get a fixed aperture distance. Only for aperture at the first surface.
        if aper_idx == 0:
            d_aper = 0.1 if self.is_cellphone else 2.0

            # If the first surface is concave, use the maximum negative sag.
            aper_r = self.surfaces[aper_idx].r
            sag1 = -self.surfaces[aper_idx + 1].surface(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in diff_surf_range:
                self.surfaces[i].d -= delta_aper

        # ==> Rule 3: If two surfaces overlap (at center), seperate them by a small distance
        for i in range(0, len(self.surfaces) - 1):
            if self.surfaces[i].d > self.surfaces[i + 1].d:
                self.surfaces[i + 1].d += 0.1
                shape_changed = True

        # ==> Rule 4: Prune all surfaces
        self.prune_surf(expand_surf=expand_surf)

        if shape_changed:
            print("Surface shape corrected.")
        return shape_changed

    # ====================================================================================
    # Visualization.
    # ====================================================================================

    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        multi_plot=False,
        plot_invalid=True,
        zmx_format=True,
        depth=DEPTH,
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
            filename=save_name,
            multi_plot=multi_plot,
            entrance_pupil=True,
            plot_invalid=plot_invalid,
            zmx_format=zmx_format,
            lens_title=lens_title,
            depth=depth,
        )

        # Draw spot diagram and PSF map
        # self.draw_psf_map(save_name=save_name, ks=101, depth=depth)

        # Calculate RMS error
        rms_avg, rms_radius_on_axis, rms_radius_off_axis = self.analysis_rms(
            depth=depth
        )
        print(
            f"Depth {depth}, on-axis RMS radius: {round(rms_radius_on_axis.item()*1000,3)}um, Off-axis RMS radius: {round(rms_radius_off_axis.item()*1000,3)}um, Avg RMS spot size (radius): {round(rms_avg.item()*1000,3)}um."
        )

        # Render an image, compute PSNR and SSIM
        if render:
            img_org = cv.cvtColor(cv.imread("./datasets/IQ/img1.png"), cv.COLOR_BGR2RGB)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=128,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                noise=0.01,
            )

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

    @torch.no_grad()
    def draw_layout(
        self,
        filename,
        depth=None,
        entrance_pupil=True,
        zmx_format=False,
        plot_invalid=True,
        multi_plot=False,
        lens_title=None,
        ax=None,
        fig=None,
    ):
        """Plot lens layout with ray tracing."""
        num_rays = 11

        # ==> Title
        if lens_title is None:
            if self.aper_idx is not None:
                fnum = self.foclen / self.entrance_pupil()[1] / 2
                lens_title = f"FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_F/{round(fnum,2)}_DIAG{round(self.r_sensor*2, 2)}mm_FocLen{round(self.foclen,2)}mm"
            else:
                lens_title = f"FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_DIAG{round(self.r_sensor*2, 2)}mm_FocLen{round(self.foclen,2)}mm"

        # ==> Plot RGB seperately
        if multi_plot:
            R = self.surfaces[0].r
            views = np.linspace(0, np.rad2deg(self.hfov) * 0.99, num=7)
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            fig, axs = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(lens_title)

            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.plot_setup2D(ax=ax, fig=fig, zmx_format=zmx_format)

                for view in views:
                    if depth is None:
                        ray = self.sample_parallel_2D(
                            R,
                            wvln,
                            view=view,
                            M=num_rays,
                            entrance_pupil=entrance_pupil,
                        )
                    else:
                        ray = self.sample_point_source_2D(
                            depth=depth,
                            view=view,
                            M=num_rays,
                            entrance_pupil=entrance_pupil,
                            wvln=wvln,
                        )

                    ps, oss = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.plot_raytraces(
                        oss,
                        ax=ax,
                        fig=fig,
                        color=colors_list[i],
                        plot_invalid=plot_invalid,
                        ra=ray.ra,
                    )
                    ax.axis("off")

            fig.savefig(f"{filename}.svg", bbox_inches="tight", format="svg", dpi=600)
            fig.savefig(f"{filename}.png", bbox_inches="tight", format="png", dpi=300)
            plt.close()

        # ==> Plot RGB in one figure
        else:
            R = self.surfaces[0].r
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            views = [0, np.rad2deg(self.hfov) * 0.707, np.rad2deg(self.hfov) * 0.99]
            ax, fig = self.plot_setup2D(zmx_format=zmx_format, ax=ax, fig=fig)

            for i, view in enumerate(views):
                if depth is None:
                    ray = self.sample_parallel_2D(
                        R,
                        WAVE_RGB[2 - i],
                        view=view,
                        M=num_rays,
                        entrance_pupil=entrance_pupil,
                    )
                else:
                    ray = self.sample_point_source_2D(
                        depth=depth,
                        view=view,
                        M=num_rays,
                        entrance_pupil=entrance_pupil,
                        wvln=WAVE_RGB[2 - i],
                    )

                ps, oss = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.plot_raytraces(
                    oss,
                    ax=ax,
                    fig=fig,
                    color=colors_list[i],
                    plot_invalid=plot_invalid,
                    ra=ray.ra,
                )

            ax.axis("off")
            ax.set_title(lens_title, fontsize=10)
            fig.savefig(f"{filename}.png", format="png", dpi=600)
            plt.close()

    def plot_raytraces(
        self,
        oss,
        ax=None,
        fig=None,
        color="b-",
        show=True,
        p=None,
        valid_p=None,
        plot_invalid=True,
        ra=None,
    ):
        """Plot ray paths."""
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D()
            show = True
        else:
            show = False

        for i, os in enumerate(oss):
            o = torch.tensor(np.array(os)).to(self.device)
            x = o[..., 0]
            z = o[..., 2]

            o = o.cpu().detach().numpy()
            z = o[..., 2].flatten()
            x = o[..., 0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i, 0])
                    z = np.append(z, p[i, 2])

            # Ignore the point before the first surface
            x = x[1:]
            z = z[1:]

            if plot_invalid:
                ax.plot(z, x, color, linewidth=0.8)
            elif ra[i] > 0:
                ax.plot(z, x, color, linewidth=0.8)

        if show:
            plt.show()
        else:
            plt.close()

        return ax, fig

    def plot_setup2D(
        self,
        ax=None,
        fig=None,
        color="k",
        linestyle="-",
        zmx_format=False,
        fix_bound=False,
    ):
        """Draw lens layout in 2D."""

        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        # Draw lens surfaces
        for i, s in enumerate(self.surfaces):
            # DOE
            if isinstance(s, Diffractive_GEO):
                s.draw_widget(ax, color="black")

            # Thin lens
            elif isinstance(s, ThinLens):
                s.draw_widget(ax, color="black")

            # Aperture
            elif isinstance(s, Aperture):
                s.draw_widget(ax, color="orange")
            
            # Lens surface
            else:
                s.draw_widget(ax, color="black")

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
    def draw_spot_diagram(self, M=7, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """Draw spot diagram of the lens. Shot rays from grid points in object space, trace to sensor and visualize."""
        # Sample and trace rays from grid points
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(
            M=M,
            R=self.sensor_size[0] / 2 / mag,
            depth=depth,
            wvln=wvln,
            spp=1024,
            pupil=True,
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
            plt.savefig(
                f"./spot{-depth}mm.png", bbox_inches="tight", format="png", dpi=300
            )
        else:
            plt.savefig(
                f"{save_name}_spot{-depth}mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )

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
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(
            M=M * 2 - 1,
            R=self.sensor_size[0] / 2 / mag,
            depth=depth,
            spp=1024,
            pupil=True,
            wvln=589.3,
        )
        ray, _, _ = self.trace(ray)
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
        relative_fov=[0.0, 0.7, 1.0],
        save_name="./mtf.png",
        wvlns=DEFAULT_WAVE,
        depth=DEPTH,
    ):
        """Draw MTF curve of the lens."""
        if save_name[-4:] != ".png":
            save_name += ".png"

        relative_fov = (
            [relative_fov] if isinstance(relative_fov, float) else relative_fov
        )
        wvlns = [wvlns] if isinstance(wvlns, float) else wvlns
        color_list = "rgb"

        plt.figure(figsize=(6, 6))
        for wvln_idx, wvln in enumerate(wvlns):
            for fov_idx, fov in enumerate(relative_fov):
                point = torch.tensor([fov, fov, depth])
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

        return

    def draw_distortion(self, depth=DEPTH, save_name=None):
        """Draw distortion."""
        # Ray tracing to calculate distortion map
        M = 15
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(
            M=M,
            spp=SPP_CALC,
            depth=depth,
            R=self.sensor_size[0] / 2 * scale,
            pupil=True,
        )
        o1 = ray.o.detach().cpu()
        x1 = o1[0, :, :, 0] / scale
        y1 = o1[0, :, :, 1] / scale

        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o2 = o2.clone().cpu()
        x2 = torch.sum(o2[..., 0] * ray.ra.cpu(), axis=0) / torch.sum(
            ray.ra.cpu(), axis=0
        )
        y2 = torch.sum(o2[..., 1] * ray.ra.cpu(), axis=0) / torch.sum(
            ray.ra.cpu(), axis=0
        )

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title("Lens distortion")
        ax.scatter(x1, y1, s=2)
        ax.scatter(x2, y2, s=2)
        ax.legend(["ref", "distortion"])
        ax.axis("scaled")

        if save_name is None:
            plt.savefig(
                f"./distortion{-depth}mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )
        else:
            plt.savefig(
                f"{save_name}_distortion{-depth}mm.png",
                bbox_inches="tight",
                format="png",
                dpi=300,
            )

    # ====================================================================================
    # Distortion
    # ====================================================================================

    @torch.no_grad()
    def unwarp(self, img, depth=DEPTH, grid=256, spp=256, crop=True):
        """Unwarp rendered images."""
        # Ray tracing to calculate distortion map
        scale = self.calc_scale_ray(depth)
        ray = self.sample_point_source(
            M=grid, spp=spp, depth=depth, R=self.sensor_size[0] / 2 * scale, pupil=True
        )
        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o_dist = (o2 * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(
            EPSILON
        ).unsqueeze(
            -1
        )  # shape (H, W, 2)

        # Reshape to [N, C, H, W], normalize to [-1, 1], then resize to img resolution [N, C, H, W]
        x_dist = F.interpolate(
            -o_dist.unsqueeze(0).unsqueeze(0)[..., 0] / self.sensor_size[1] * 2,
            img.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        y_dist = F.interpolate(
            o_dist.unsqueeze(0).unsqueeze(0)[..., 1] / self.sensor_size[0] * 2,
            img.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        grid_dist = torch.stack((x_dist.squeeze(0), y_dist.squeeze(0)), dim=-1)

        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(img, grid_dist, align_corners=True)

        return img_unwarpped

    # ====================================================================================
    # Loss function
    # ====================================================================================

    def loss_infocus(self, bound=0.005):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            bound (float, optional): bound of RMS loss. Defaults to 0.005 [mm].
        """
        focz = self.d_sensor
        loss = []
        for wv in WAVE_RGB:
            # Ray tracing
            ray = self.sample_parallel(fov=0.0, M=31, wvln=wv, entrance_pupil=True)
            ray, _, _ = self.trace(ray)
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
        # H, W = self.sensor_res
        H = 31

        # ==> PSF and RMS by patch
        scale = -depth * np.tan(self.hfov) / self.r_sensor

        rms = 0.0
        for wvln in WAVE_RGB:
            ray = self.sample_point_source(
                M=H,
                spp=SPP_PSF,
                depth=depth,
                R=self.sensor_size[0] / 2 * scale,
                pupil=True,
                wvln=wvln,
            )
            ray, _, _ = self.trace(ray)
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
        ray = self.sample_point_source_2D(
            depth=depth, view=self.hfov * 57.3, M=7, entrance_pupil=True
        )
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
        """Loss function to avoid self-intersection. Loss is designed by the distance to the next surfaces.
        """
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
            z_front = current_surf.surface(r, 0) + current_surf.d
            z_next = next_surf.surface(r, 0) + next_surf.d
            
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
        z_last_surf = self.d_sensor - last_surf.surface(r, 0) - last_surf.d
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
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(
            M=M, spp=spp, depth=DEPTH, R=scale * self.sensor_size[0] / 2, pupil=True
        )

        # Ray tracing
        ray, _, _ = self.trace(ray)

        # Loss (we want to maximize ray angle term)
        loss = ray.obliq.min()
        loss = min(loss, target)

        return -loss

    def loss_reg(self, w_focus=None):
        """An empirical regularization loss for lens design.

        By default we should use weight 0.1 * self.loss_reg()
        """
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

    # ====================================================================================
    # Optimization
    # ====================================================================================
    def get_optimizer_params(
        self,
        lr=[1e-4, 1e-4, 1e-1, 1e-3],
        decay=0.01,
        diff_surf_range=None,
        optim_mat=False,
    ):
        """Get optimizer parameters for different lens surface.

            For cellphone lens: [c, d, k, a], [1e-4, 1e-4, 1e-1, 1e-4]
            For camera lens: [c, d, 0, 0], [1e-3, 1e-4, 0, 0]

        Args:
            lr (list): learning rate for different parameters [c, d, k, a]. Defaults to [1e-4, 1e-4, 0, 1e-4].
            decay (float): decay rate for higher order a. Defaults to 0.2.
            diff_surf_range (list): surface indices to be optimized. Defaults to None.

        Returns:
            list: optimizer parameters
        """
        diff_surf_range = (
            self.diff_surf_range if diff_surf_range is None else diff_surf_range
        )
        params = []
        for i in diff_surf_range:
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

            else:
                raise Exception(f"Surface type {surf.__class__.__name__} is not supported for optimization yet.")

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
        importance_sampling=False,
        optim_mat=False,
        match_mat=False,
        shape_control=True,
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

        # Training
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
                        zmx_format=True,
                        plot_invalid=True,
                        multi_plot=False,
                    )

            # ===> Sample new rays and calculate center
            if i % sample_rays_per_iter == 0:
                with torch.no_grad():
                    # Sample rays
                    scale = self.calc_scale_pinhole(depth)
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.sample_point_source(
                            M=num_grid,
                            R=self.sensor_size[0] / 2 * scale,
                            depth=depth,
                            spp=spp,
                            pupil=True,
                            wvln=wv,
                            importance_sampling=importance_sampling,
                        )
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_p = -self.psf_center(
                            point=ray.o[0, ...], method="chief_ray"
                        )
                    else:
                        center_p = -self.psf_center(
                            point=ray.o[0, ...], method="pinhole"
                        )

            # ===> Optimize lens by minimizing RMS
            loss_rms = []
            for j, wv in enumerate(WAVE_RGB):
                # Ray tracing
                ray = rays_backup[j].clone()
                ray, _, _ = self.trace(ray)
                ra = ray.ra.clone()
                xy = ray.project_to(self.d_sensor)
                xy_norm = (xy - center_p) * ray.ra.unsqueeze(-1)

                # Use only quater of the sensor
                xy_norm = xy_norm[:, num_grid // 2 :, num_grid // 2 :]
                ra = ra[:, num_grid // 2 :, num_grid // 2 :]

                # Weight mask
                weight_mask = (xy_norm.clone().detach() ** 2).sum([0, -1]) / (
                    ra.sum([0]) + EPSILON
                )  # Use L2 error as weight mask
                weight_mask /= weight_mask.mean()  # shape of [M, M]

                # RMS loss
                l_rms = torch.sqrt(
                    torch.sum((xy_norm**2 + EPSILON).sum(-1) * weight_mask)
                    / (torch.sum(ra) + EPSILON)
                )  # Weighted L2 loss
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

    # ====================================================================================
    # Lesn file IO
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

                else:
                    raise Exception(f"Surface type {surf_dict['type']} is not implemented in GeoLens.read_lens_json().")

                self.surfaces.append(s)
                d += surf_dict["d_next"]

        # self.sensor_size = data['sensor_size']
        self.sensor_res = data["sensor_res"] if "sensor_res" in data else [1024, 1024]
        self.r_sensor = data["r_sensor"]
        self.d_sensor = torch.tensor(d)
        self.lens_info = data["info"] if "info" in data else "None"

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

        # Head string
        head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {self.surfaces[0].r*2}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707*self.hfov*57.3} {0.99*self.hfov*57.3}
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
        sensor_str = f"""SURF {i+2}
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
    imgh = 2 * foclen * np.tan(fov / 2 / 57.3)
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
            d_next = (torch.rand(1) + 0.5).item()  # Replace numpy with torch
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
    lens.find_aperture()
    lens.prepare_sensor(
        sensor_res=lens.sensor_res,
        sensor_size=[imgh / math.sqrt(2), imgh / math.sqrt(2)],
    )
    lens.diff_surf_range = lens.find_diff_surf()
    lens.post_computation()

    # Save lens
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}.json"
    lens.write_lens_json(os.path.join(save_dir, filename))

    return lens


def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """Create a surface object based on the surface type."""
    c = np.random.randn(1).astype(np.float32) * 0.001
    r = max(imgh / 2, aper_r)

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)
    elif surface_type == "Aspheric":
        ai = np.random.randn(7).astype(np.float32) * 1e-30
        k = np.random.randn(1).astype(np.float32) * 0.001
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)
    else:
        raise Exception("Surface type not supported yet.")

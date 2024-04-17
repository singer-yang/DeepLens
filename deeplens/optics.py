""" Lensgroup class. Use geometric ray tracing to optical computation.

Technical Paper:
Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint (2023)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import torch
import random
import json
import cv2 as cv
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from transformers import get_cosine_schedule_with_warmup

from .surfaces import *
from .utils import *
from .monte_carlo import *
from .render_psf import *
from .basics import GEO_SPP, EPSILON, WAVE_SPEC

class Lensgroup(DeepObj):
    def __init__(self, filename=None, sensor_res=[1024, 1024], use_roc=False, device=DEVICE):
        """ Initialize Lensgroup.

        Args:
            filename (string): lens file.
            device ('cpu' or 'cuda'): device.
            sensor_res: (H, W)
        """
        super(Lensgroup, self).__init__()
        
        # Load lens file.
        if filename is not None:
            self.lens_name = filename
            self.device = device
            self.load_file(filename, use_roc, sensor_res)
            self.to(device)

            # Lens calculation
            self.find_aperture()
            self.prepare_sensor(sensor_res, sensor_size=self.sensor_size)
            self.diff_surf_range = self.find_diff_surf()
            self.post_computation()         
        
        else:
            self.sensor_res = sensor_res
            self.surfaces = []
            self.materials = []


    def load_file(self, filename, use_roc, sensor_res):
        """ Load lens from .txt file.

        Args:
            filename (string): lens file.
            use_roc (bool): use radius of curvature (roc) or not. In the old code, we store lens data in roc rather than curvature.
            post_computation (bool): compute fnum, fov, foclen or not.
            sensor_res (list): sensor resolution.
        """
        if filename[-4:] == '.txt':
            self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(filename, use_roc)
            self.d_sensor = d_last + self.surfaces[-1].d.item()
            self.sensor_size = [2 * self.r_last * sensor_res[0] / np.sqrt(sensor_res[0]**2 + sensor_res[1]**2), 2 * self.r_last * sensor_res[1] / np.sqrt(sensor_res[0]**2 + sensor_res[1]**2)]
            self.focz = self.d_sensor
        
        elif filename[-5:] == '.json':
            self.read_lens_json(filename)

        else:
            raise Exception("File format not supported.")

    def load_external(self, surfaces, materials, r_last, d_sensor):
        """ Load lens from extrenal surface/material list.
        """
        self.surfaces = surfaces
        self.materials = materials
        self.r_last = r_last
        self.d_sensor = d_sensor

        for i in range(len(self.surfaces)):
            self.surfaces[i].mat1 = self.materials[i]
            self.surfaces[i].mat2 = self.materials[i+1]

    def prepare_sensor(self, sensor_res=[512, 512], sensor_size=None):
        """ Create sensor. 

            reference values:
                Nikon z35 f1.8: diameter = 1.912 [cm] ==> But can we just use [mm] in our code?
                Congli's caustic example: diameter = 12.7 [mm]
        Args:
            sensor_res (list): Resolution, pixel number.
            pixel_size (float): Pixel size in [mm].

            sensor_res: (H, W)
        """
        sensor_res = [sensor_res, sensor_res] if isinstance(sensor_res, int) else sensor_res
        self.sensor_res = sensor_res
        H, W = sensor_res
        if sensor_size is None:
            self.sensor_size = [2 * self.r_last * H / np.sqrt(H**2 + W**2), 2 * self.r_last * W / np.sqrt(H**2 + W**2)]
        else:
            self.sensor_size = sensor_size
            self.r_last = np.sqrt(sensor_size[0]**2 + sensor_size[1]**2) / 2

        assert self.sensor_size[0] / self.sensor_size[1] == H / W, "Pixel is not square."
        self.pixel_size = self.sensor_size[0] / sensor_res[0]


    def post_computation(self):
        """ After loading lens, compute foclen, fov and fnum.
        """
        self.find_aperture()
        self.hfov = self.calc_fov()
        self.foclen = self.calc_efl()
        
        avg_pupilz, avg_pupilx = self.entrance_pupil()
        self.fnum = self.foclen / avg_pupilx / 2

        if self.r_last < 8.0:
            self.is_cellphone = True


    def find_aperture(self):
        """ Find aperture. If the lens has no aperture, use the surface with the smallest radius.
        """
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Aperture):
                self.aper_idx = i
                return

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])

    def find_diff_surf(self):
        """ Get surface indices without aperture.
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(range(self.aper_idx+1, len(self.surfaces)))
        return diff_surf_range

    # ====================================================================================
    # Ray Sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(self, R=None, wvln=DEFAULT_WAVE, z=None, view=0.0, M=15, forward=True, entrance_pupil=False):
        """ Sample 2D parallel rays. Rays have shape [M, 3].
        
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

            x2 = torch.linspace(- pupilx, pupilx, M) * 0.99
            y2 = torch.zeros_like(x2)
            z2 = torch.full_like(x2, pupilz)
            o2 = torch.stack((x2,y2,z2), axis=-1)   # shape [M, 3]
            
            dx = torch.full_like(x2, np.sin(view / 57.3))
            dy = torch.zeros_like(x2)
            dz = torch.full_like(x2, np.cos(view / 57.3))
            d = torch.stack((dx,dy,dz), axis=-1)

            # Move ray origins to z = -0.1 for tracing
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
                z = 0 if forward else self.d_sensor
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

            d = torch.stack((dx,dy,dz), axis=-1)

            return Ray(o, d, wvln, device=self.device)


    @torch.no_grad()
    def sample_parallel(self,fov=0.0, R=None, z=None, M=15,  wvln=DEFAULT_WAVE, sampling='grid', forward=True, entrance_pupil=False):
        """ Sample parallel rays from plane (-R:R, -R:R, z). Rays have shape [spp, M, M, 3]
        
            Used for (1) in-focus loss, (2) RMS spot radius calculation

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
        fov = np.radians(np.asarray(fov))   # convert degree to radian
        
        # Sample ray origins
        if entrance_pupil:
            pupilz, pupilr = self.entrance_pupil()
            if sampling == 'grid': # sample a square 
                x, y = torch.meshgrid(
                    torch.linspace(-pupilr, pupilr, M),
                    torch.linspace(pupilr, -pupilr, M),
                    indexing='xy'
                )
            elif sampling == 'radial':
                r2 = torch.rand((M, M)) * pupilr**2
                theta = torch.rand((M, M)) * 2 * np.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception('Sampling method not implemented!')

        else:
            if R is None:
                # We want to sample at a depth, so radius of the cone need to be computed.
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0).item() # sag is a float
                R = np.tan(fov) * sag + self.surfaces[0].r 

            if sampling == 'grid': # sample a square 
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M),
                    torch.linspace(R, -R, M),
                    indexing='xy'
                )
            elif sampling == 'radial':
                r2 = torch.rand((M, M)) * R**2
                theta = torch.rand((M, M)) * 2 * np.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception('Sampling method not implemented!')

        # Generate rays
        if isinstance(fov, float):
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2)
            d = torch.zeros_like(o)
            if forward:
                d[...,2] = torch.full_like(x, np.cos(fov))
                d[...,0] = torch.full_like(x, np.sin(fov))
            else:
                d[...,2] = torch.full_like(x, -np.cos(fov))
                d[...,0] = torch.full_like(x, -np.sin(fov))
        else:
            spp = len(fov)
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2).unsqueeze(0).repeat(spp, 1, 1, 1)
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
    def sample_point_source_2D(self, depth=-1000, view=0, M=7, entrance_pupil=False, wvln=DEFAULT_WAVE):
        """ Sample point source 2D rays. Rays hape shape of [M, 3].

            Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to given depth

        Args:
            depth (float, optional): sampling depth. Defaults to -1000.
            view (float, optional): incident angle (in degree). Defaults to 0.
            M (int, optional): ray number. Defaults to 9.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        """
        if entrance_pupil:
            pupilz, pupilx = self.entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        # Second point on the pupil or first surface
        x2 = torch.linspace(-pupilx, pupilx, M, device=self.device) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), axis=1)

        # First point is the point source
        o1 = torch.zeros_like(o2)
        o1[:, 2] = depth
        o1[:, 0] = depth * np.tan(view / 57.3)

        # Form the rays and propagate to z = 0
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=self.device)
        ray.propagate_to(z=self.surfaces[0].d - 0.1)    # ray starts from z = - 0.1

        return ray


    @torch.no_grad()
    def sample_point_source(self, R=None, depth=-10.0, M=11, spp=16, fov=10.0, forward=True, pupil=True, wvln=DEFAULT_WAVE, importance_sampling=False):
        """ Sample forward point-grid rays. Rays have shape [spp, M, M, 3]
        
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
        """
        if R is None:
            R = self.surfaces[0].r
        Rw = R * self.sensor_res[1] / self.sensor_res[0] # half height

        # sample o
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, M),
            torch.linspace(1, -1, M),
            indexing='xy'
            )

        if importance_sampling:
            x = torch.sqrt(x.abs()) * x.sign()
            y = torch.sqrt(y.abs()) * y.sign()

        x = x * Rw
        y = y * R

        # x, y = torch.t(x), torch.t(y)
        z = torch.full_like(x, depth)
        o = torch.stack((x,y,z), -1).to(self.device)
        o = o.unsqueeze(0).repeat(spp, 1, 1, 1)
        
        # sample d
        o2 = self.sample_pupil(res=(M,M), spp=spp)
        d = o2 - o
        d = d / torch.linalg.vector_norm(d, ord=2, dim=-1, keepdim=True)

        # generate ray
        ray = Ray(o, d, wvln, device=self.device)
        return ray


    @torch.no_grad()
    def sample_from_points(self, o=[[0., 0., -10000.]], spp=256, wvln=DEFAULT_WAVE, shrink_pupil=False):
        """ Sample forward rays from given point source (un-normalized positions). Rays have shape [spp, N, 3]

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
        # Compute o, shape [spp, N, 3]
        if not torch.is_tensor(o):
            o = torch.tensor(o)
        o = o.unsqueeze(0).repeat(spp, 1, 1)
        
        # Sample pupil and compute d
        pupilz, pupilr = self.entrance_pupil(shrink_pupil=shrink_pupil)
        theta = torch.rand(spp) * 2 * np.pi
        r = torch.sqrt(torch.rand(spp)*pupilr**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), 1)
            
        d = o2.unsqueeze(1) - o
        
        # Calculate rays
        ray = Ray(o, d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_sensor(self, spp=64, pupil=True, wvln=DEFAULT_WAVE, sub_pixel=False):
        """ Sample rays from sensor pixels. Rays have shape of [spp, H, W, 3].

        Args:
            sensor_scale (int, optional): number of pixels remain the same, but only sample rays on part of sensor. Defaults to 1.
            spp (int, optional): sample per pixel. Defaults to 1.
            vpp (int, optional): sample per pixel on pupil. Defaults to 64.
            high_spp (bool, optional): whether to use high spp. Defaults to False.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        """
        # ===> sample o1 on sensor plane
        # In 'render_compute_img' func, we use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(-self.sensor_size[1]/2, self.sensor_size[1]/2, self.sensor_res[1]+1, device=self.device)[1:],
            torch.linspace(self.sensor_size[0]/2, -self.sensor_size[0]/2, self.sensor_res[0]+1, device=self.device)[1:],
            indexing='xy'
        )
        z1 = torch.full_like(x1, self.d_sensor, device= self.device)

        # ==> Sample o2 on the second plane and compute rays
        pupilz, pupilr = self.exit_pupil()

        # ==> Use bottom-right corner to represent each pixel
        o2 = self.sample_pupil(self.sensor_res, spp, pupilr=pupilr, pupilz=pupilz)

        o = torch.stack((x1, y1, z1), 2)
        o = torch.broadcast_to(o, o2.shape)
        d = o2 - o    # broadcast to [spp, H, W, 3]
            
        ray = Ray(o, d, wvln, device=self.device)
        return ray


    @torch.no_grad()
    def sample_pupil(self, res=(512,512), spp=16, num_angle=8, pupilr=None, pupilz=None):
        """ Sample points (not rays) on the pupil plane with rings. Points have shape [spp, res, res].

            2*pi is devided into [num_angle] sectors.
            Circle is devided into [spp//num_angle] rings.

        Args:
            res (tuple): pupil plane resolution. Defaults to (512,512).
            spp (int): sample per pixel. Defaults to 16.
            num_angle (int): number of sectors. Defaults to 8.
            pupilr (float): pupil radius. Defaults to None.
            pupilz (float): pupil z position. Defaults to None.
            multiplexing (bool): whether to use multiplexing. Defaults to False.
        """
        H, W = res
        if pupilr is None or pupilz is None:
            pupilz, pupilr = self.entrance_pupil()

        # => Naive implementation
        if spp % num_angle != 0 or spp >= 10000:
            theta = torch.rand((spp, H, W), device=self.device) * 2 * np.pi
            r2 = torch.rand((spp, H, W), device=self.device) * pupilr**2
            r = torch.sqrt(r2)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        # => Sample more uniformly when spp is not large
        else:
            num_r2 = spp // num_angle
            
            # ==> For each pixel, sample different points on the pupil
            x, y = [], []
            for i in range(num_angle):
                for j in range(spp//num_angle):
                    delta_theta = torch.rand((1, *res), device=self.device) * 2 * np.pi / num_angle # sample delta_theta from [0, pi/4)
                    theta = delta_theta + i * 2 * np.pi / num_angle 

                    delta_r2 = torch.rand((1, *res), device=self.device) * pupilr**2 / spp * num_angle
                    r2 = delta_r2 + j * pupilr**2 / spp * num_angle
                    r = torch.sqrt(r2)

                    x.append(r * torch.cos(theta))
                    y.append(r * torch.sin(theta))
            
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        return o



    # ====================================================================================
    # Ray Tracing functions
    # ====================================================================================
    def trace(self, ray, lens_range=None, record=False):
        """ General ray tracing function. 

        Args:
            ray ([type]): [description]
            stop_ind ([int]): Early stop index.
            record: Only when we want to plot ray path, set `record` to True.

        Returns:
            ray_final (Ray object): ray after optical system.
            valid (boolean matrix): mask denoting valid rays.
            oss (): position of ray on the sensor plane.
        """
        is_forward = (ray.d.reshape(-1,3)[0,2] > 0)
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))
        
        if is_forward:
            ray.propagate_to(self.surfaces[0].d - 0.1)    # for high-precision opd calculation
            valid, ray_out, oss = self._forward_tracing(ray, lens_range, record=record)
        else:
            valid, ray_out, oss = self._backward_tracing(ray, lens_range, record=record)

        return ray_out, valid, oss


    def trace2obj(self, ray, depth=DEPTH):
        """ Trace rays through the lens and reach the sensor plane.
        """
        ray, _, _, = self.trace(ray)
        ray = ray.propagate_to(depth)
        return ray

    
    def trace2sensor(self, ray, record=False, ignore_invalid=False):
        """ Trace optical rays to sensor plane.
        """
        if record:
            ray_out, valid, oss = self.trace(ray, record=record)
            ray_out = ray_out.propagate_to(self.d_sensor)
            valid = (ray_out.ra == 1)
            p = ray.o
            for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                if v.any():
                    os.append(pp)

            if ignore_invalid:
                p = p[valid]
            else:
                assert len(p.shape) >= 2, 'This function is not tested.'
                p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))

            for v, os, pp in zip(valid, oss, p):
                if v:
                    os.append(pp.cpu().detach().numpy())
            return p, oss

        else:
            ray, _, _, = self.trace(ray)
            ray = ray.propagate_to(self.d_sensor)
            return ray


    def _forward_tracing(self, ray, lens_range, record):
        """ Trace rays from object space to sensor plane.
        """
        dim = ray.o[..., 2].shape

        if record:
            oss = []    # oss records all points of intersection. ray.o shape of [N, 3]
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in lens_range:
            ray = self.surfaces[i].ray_reaction(ray)
            
            valid = (ray.ra == 1)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)
        
        valid = (ray.ra == 1)
        return valid, ray, oss


    def _backward_tracing(self, ray, lens_range, record):
        """ Trace rays from sensor plane to object space.
        """
        dim = ray.o[..., 2].shape
        valid = (ray.ra == 1)
        
        if record:
            oss = []    # oss records all points of intersection
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in np.flip(lens_range):
            ray = self.surfaces[i].ray_reaction(ray)

            valid = (ray.ra > 0)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)

        valid = (ray.ra == 1)
        return valid, ray, oss


        
    # ====================================================================================
    # Ray-tracing based rendering
    # ====================================================================================
    @torch.no_grad()
    def render_single_img(self, img_org, depth=DEPTH, spp=64, unwarp=False, save_name=None, return_tensor=False, noise=0, method='raytracing'):
        """ Render a single image for visualization and debugging.

            This function is designed non-differentiable. If want to use differentiable rendering, call self.render() function.

        Args:
            img_org (ndarray): ndarray read by opencv.
            render_unwarp (bool, optional): _description_. Defaults to False.
            depth (float, optional): _description_. Defaults to DEPTH.
            save_name (string, optional): _description_. Defaults to None.

        Returns:
            ing_render (ndarray): rendered image. uint8 dtype and ndarray.
        """
        if not isinstance(img_org, np.ndarray):
            raise Exception('This function only supports ndarray input. If you want to render an image batch, use `render` function.')

        # ==> Prepare sensor to match the image resolution
        sensor_res = self.sensor_res
        if len(img_org.shape) == 2:
            rgb = False
            H, W = img_org.shape
            raise Exception('Monochrome image is not tested yet.')
        elif len(img_org.shape) == 3:
            rgb = True 
            H, W, C = img_org.shape
            assert C == 3, 'Only support RGB image, dtype should be ndarray.'
        self.prepare_sensor(sensor_res=[H, W])

        img = torch.tensor((img_org/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if method == 'raytracing':
            # ==> Render object image by ray-tracing
            scale = self.calc_scale_ray(depth=depth)
            img = torch.flip(img, [-2, -1])
            if rgb:
                img_render = torch.zeros_like(img)
                # Normal rendering
                if spp <= 64:
                    for i in range(3):
                        ray = self.render_sample_ray(spp=spp, wvln=WAVE_RGB[i])
                        ray, _, _ = self.trace(ray) 
                        img_render[:,i,:,:] = self.render_compute_image(img[:,i,:,:], depth, scale, ray)
                # High-spp rendering
                else:
                    iter_num = int(spp // 64)
                    for ii in range(iter_num):
                        for i in range(3):
                            ray = self.render_sample_ray(spp=64, wvln=WAVE_RGB[i])
                            ray, _, _ = self.trace(ray) 
                            img_render[:,i,:,:] += self.render_compute_image(img[:,i,:,:], depth, scale, ray, train=False)
                    img_render /= iter_num
            else:
                ray = self.render_sample_ray(spp=spp, wvln=DEFAULT_WAVE)
                ray, _, _ = self.trace(ray)
                img_render = self.render_compute_image(img, depth, scale, ray)
        
        elif method == 'psf':
            psf_grid = 7
            psf_ks = 21
            psf_map = self.psf_map(grid=psf_grid, ks=psf_ks, depth=depth)
            img_render = render_psf_map(img, psf_map, grid=psf_grid)
        
        # ==> Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

        # ==> Add noise
        if noise > 0:
            img_render = img_render + torch.randn_like(img_render) * noise
            img_render = torch.clamp(img_render, 0, 1)
        

        if save_name is not None:
            save_image(img_render, f'{save_name}.png')

        # ==> Change the sensor resolution back
        self.prepare_sensor(sensor_res=sensor_res)

        if return_tensor:
            return img_render
        else:
            # ==> Convert to uint8
            img_render = img_render[0,...].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            return img_render


    def render(self, img, depth=DEPTH, spp=64, psf_grid=9, psf_ks=101, method='ray_tracing'):
        """ This function is defined for End-to-End lens design. It simulates the camera-captured image batch and it is differentiable.

            2 kinds of rendering methods are supported:
                1. ray tracing based rendering
                2. PSF based rendering

        Args:
            img (tensor): [N, C, H, W] shape image batch.
            depth (float, optional): depth of object image. Defaults to DEPTH.
            spp (int, optional): sample per pixel. Defaults to 64.
            grid (int, optional): psf grid number. Defaults to 9.
            method (str, optional): rendering method. Defaults to 'ray_tracing'.

        Returns:
            img_render (tensor): [N, C, H, W] shape rendered image batch.
        """
        if method == 'ray_tracing':
            img = torch.flip(img, [-2, -1])
            scale = self.calc_scale_pinhole(depth=depth)

            img_render = torch.zeros_like(img)
            for i in range(3):
                ray = self.render_sample_ray(spp=spp, wvln=WAVE_RGB[i])
                ray, _, _ = self.trace(ray) 
                img_render[:,i,:,:] = self.render_compute_image(img[:,i,:,:], depth, scale, ray)
        
        elif method == 'psf':
            # Note: larger psf_grid and psf_ks are better
            psf_map = self.psf_map(grid=psf_grid, ks=psf_ks, depth=depth)
            img_render = render_psf_map(img, psf_map, grid=psf_grid)
        
        else:
            raise Exception('Unknown method.')
        
        return img_render

    def render_sample_ray(self, spp=64, wvln=DEFAULT_WAVE):
        """ Ray tracing rendering step1: sample ray and go through lens.
        """
        ray = self.sample_sensor(spp=spp, pupil=True, wvln=wvln)
        return ray


    def render_compute_image(self, img, depth, scale, ray, point_pixel=True, train=True, noise=0, coherent=False):
        """ Ray tracing rendering step2: ray and texture plane intersection and computer rendered image.

            With interpolation. Can either receive tensor or ndarray

            This function receives [spp, W, H, 3] shape ray, returns [W, H, 3] shape sensor output.

            backpropagation, I -> w_i -> u -> p -> ray
        """
        # ====> Preparetion
        if torch.is_tensor(img):    # if img is [N, C, H, W] or [N, H, W] tensor, what situation will [N, H, W] occur?
            H, W = img.shape[-2:]
            if len(img.shape) == 4:
                img = F.pad(img, (1,1,1,1), "replicate")    # we MUST use replicate padding.
            else:
                img = F.pad(img.unsqueeze(1), (1,1,1,1), 'replicate').squeeze(1)

        elif isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img / 255.0
                img = img.astype(np.float32)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            H, W = img.shape[:2]
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
            img = F.pad(img, (1,1,1,1), "replicate")

        # ====> Scale scene image to get 1:1 alignment.
        ray = ray.propagate_to(depth)
        p = ray.o[...,:2]
        pixel_size = scale * self.pixel_size
        ray.ra = ray.ra * (torch.abs(p[...,0]/pixel_size) < (W/2+1)) * (torch.abs(p[...,1]/pixel_size) < (H/2+1))
        
        # ====> Monte carlo integral
        # Convert to uv coordinates
        u = torch.clamp(W/2 + p[..., 0]/pixel_size, min=-0.99, max=W-0.01)
        v = torch.clamp(H/2 + p[..., 1]/pixel_size, min=0.01, max=H+0.99) 

        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1

        # Gradients are stored in weight parameters
        w_i = v - v.floor().long()
        w_j = u.ceil().long() - u

        # Bilinear interpolation
        # img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], irr_img shape [N, C, spp, H, W]
        irr_img =  img[...,idx_i, idx_j] * w_i * w_j
        irr_img += img[...,idx_i+1, idx_j] * (1-w_i) * w_j
        irr_img += img[...,idx_i, idx_j+1] * w_i * (1-w_j)
        irr_img += img[...,idx_i+1, idx_j+1] * (1-w_i) * (1-w_j)

        I = (torch.sum(irr_img * ray.ra, -3) + 1e-9) / (torch.sum(ray.ra, -3) + 1e-6)   # w/ vignetting correction 
        # I = (torch.sum(irr_img * ray.ra, -3) + 1e-9) / ray.ra.shape[-3]   # w/o vignetting correction

        # ====> Add sensor noise
        if noise > 0:
            I += noise * torch.randn_like(I).to(self.device)
        
        return I


    # ====================================================================================
    # PSF and spot diagram
    #   1. Incoherent functions
    #   2. Coherent functions 
    # ====================================================================================
    def point_source_grid(self, depth, grid=9, normalized=True, quater=False, center=False):
        """ Compute point grid [-1: 1] * [-1: 1] in the object space to compute PSF grid.

        Args:
            depth (float): Depth of the point source plane.
            grid (int): Grid size. Defaults to 9.
            normalized (bool): Whether to use normalized x, y corrdinates [-1, 1]. Defaults to True.
            quater (bool): Whether to use quater of the grid. Defaults to False.
            center (bool): Whether to use center of each patch. Defaults to False.

        Returns:
            point_source: Shape of [grid, grid, 3].
        """
        if grid == 1:
            x, y = torch.tensor([[0.]]), torch.tensor([[0.]])
            assert not quater, 'Quater should be False when grid is 1.'
        else:
            # ==> Use center of each patch
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x, y = torch.meshgrid(
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid), 
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid),
                    indexing='xy')
            # ==> Use corner
            else:   
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid), 
                    torch.linspace(0.98, -0.98, grid),
                    indexing='xy')
        
        z = torch.full((grid, grid), depth)
        point_source = torch.stack([x, y, z], dim=-1)
        
        # ==> Use quater of the sensor plane to save memory
        if quater:
            z = torch.full((grid, grid), depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid // 2 if grid % 2 == 0 else grid // 2 + 1
            bound_j = grid // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        if not normalized:
            scale = self.calc_scale_pinhole(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    
    def point_source_radial(self, depth, grid=9, center=False):
        """ Compute point radial [0, 1] in the object space to compute PSF grid.

        Args:
            grid (int, optional): Grid size. Defaults to 9.

        Returns:
            point_source: Shape of [grid, 3].
        """
        if grid == 1:
            x = torch.tensor([0.])
        else:
            # Select center of bin to calculate PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x = torch.linspace(0, 1 - half_bin_size, grid)
            else:   
                x = torch.linspace(0, 0.98, grid)
        
        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source

    
    @torch.no_grad()
    def psf_center(self, point, method='chief_ray'):
        """ Compute reference PSF center (flipped, green light) for given point source.

        Args:
            point: [N, 3] un-normalized point is in object plane.

        Returns:
            psf_center: [N, 2] un-normalized psf center in sensor plane.
        """
        if method == 'chief_ray':
            # Shrink the pupil and calculate centroid ray as the chief ray. Distortion is allowed.
            ray = self.sample_from_points(point, spp=GEO_SPP, shrink_pupil=True)
            ray = self.trace2sensor(ray)
            assert (ray.ra == 1).any(), 'No sampled rays is valid.'
            psf_center = (ray.o * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON) # shape [N, 3]
            psf_center = - psf_center[..., :2]   # shape [N, 2]

        elif method == 'pinhole':
            # Pinhole camera perspective projection. This doesnot allow distortion.
            scale = self.calc_scale_pinhole(point[..., 2])
            psf_center = - point[..., :2] / scale
        
        else:
            raise Exception('Unsupported method.')

        return psf_center

    
    def psf(self, points, ks=31, wvln=DEFAULT_WAVE, spp=GEO_SPP, center=True):
        """ Single wvln incoherent PSF calculation.

        Args:
            points (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 7.
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
        point_obj = points.clone()
        point_obj[..., 0] = points[..., 0] * scale * self.sensor_size[1] / 2   # x coordinate
        point_obj[..., 1] = points[..., 1] * scale * self.sensor_size[0] / 2   # y coordinate
        
        # Trace rays to sensor plane
        ray = self.sample_from_points(o=point_obj, spp=spp, wvln=wvln)
        ray = self.trace2sensor(ray)

        # Calculate PSF
        if center:
            # PSF center on the sensor plane by chief ray
            pointc_chief_ray = self.psf_center(point_obj)   # shape [N, 2]
            psf = forward_integral(ray, ps=self.pixel_size, ks=ks, pointc_ref=pointc_chief_ray)
        else:
            # PSF center on the sensor plane by pespective
            pointc_ideal = points.clone()[:,:2]
            pointc_ideal[:, 0] *= self.sensor_size[1] / 2
            pointc_ideal[:, 1] *= self.sensor_size[0] / 2
            psf = forward_integral(ray, ps=self.pixel_size, ks=ks, pointc_ref=pointc_ideal)
        
        # Normalize to 1
        psf = psf / psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        
        if single_point:
            psf = psf.squeeze(0)

        return psf
    

    def psf_rgb(self, points, ks=31, spp=GEO_SPP, center=True):
        """ Compute RGB point PSF. This function is differentiable.
        
        Args:
            point (Tensor): Shape of [N, 3], point is in object space, normalized.
            ks (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. Defaults to 2048.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            psf: Shape of [N, 3, ks, ks] or [3, ks, ks].
        """
        psfs = []
        for wvln in WAVE_RGB:
            psfs.append(self.psf(points=points, wvln=wvln, ks=ks, spp=spp, center=center))
        
        psf = torch.stack(psfs, dim = -3)   # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf
    

    def psf_map(self, depth=DEPTH, grid=7, ks=51, spp=GEO_SPP, center=True):
        """ Compute RGB PSF map at a given depth.

            Now used for (1) rendering, (2) draw PSF map

        Args:
            grid (int, optional): Grid size. Defaults to 7.
            ks (int, optional): Kernel size. Defaults to 51.
            depth (float, optional): Depth of the point source plane. Defaults to DEPTH.
            center (bool, optional): Use spot center as PSF center. Defaults to True.
            spp (int, optional): Sample per pixel. Defaults to None.

        Returns:
            psf_map: Shape of [3, grid*ks, grid*ks].
        """
        points = self.point_source_grid(depth=depth, grid=grid, quater=False)
        points = points.reshape(-1, 3)
        psfs = self.psf_rgb(points=points, ks=ks, center=center, spp=spp) # shape [grid**2, 3, ks, ks]

        psf_map = make_grid(psfs, nrow=grid, padding=0, pad_value=0.0)
        return psf_map
    

    def psf_coherent(self, point, ks=101, wvln=DEFAULT_WAVE, spp=COHERENT_SPP, center=True, full_res=False):
        pass

    def psf2mtf(self, psf, diag=False):
        """ Convert 2D PSF kernel to MTF curve by FFT.

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

        delta_x = self.pixel_size #/ 2

        # Create frequency axis in cycles/mm
        freq = np.fft.fftfreq(psf.shape[0], delta_x)

        # Only keep the positive frequencies
        positive_freq_idx = freq > 0

        return freq[positive_freq_idx], tangential_mtf[positive_freq_idx], sagittal_mtf[positive_freq_idx]



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
        """ Calculate the focus length.
        """
        if self.r_last < 8: # Cellphone lens, we usually use EFL to describe the lens.
            return self.calc_efl()
        else:   # Camera lens, we use the to describe the lens.
            return self.calc_bfl()

    def calc_bfl(self, wvln=DEFAULT_WAVE):
        """ Compute back focal length (BFL). 

            BFL: Distance from the second principal point to in-focus position.
        """
        M = GEO_GRID

        # Forward ray tracing
        ray = self.sample_parallel_2D(R=self.surfaces[0].r * 0.1, M=M, forward=True, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        # Principal point
        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z_principal = out_ray.o[..., 2] - out_ray.d[..., 2] * t

        # Focal point
        t = - out_ray.o[..., 0] / out_ray.d[..., 0]
        z_focus = out_ray.o[..., 2] + out_ray.d[..., 2] * t

        # Back focal length
        bfl = z_focus - z_principal
        bfl = np.nanmean(bfl[ray.ra > 0].cpu().numpy())

        return bfl

    def calc_efl(self):
        """ Compute effective focal length (EFL). Effctive focal length is also commonly used to compute F/#.

            EFL: Defined by FoV and sensor radius.
        """
        return self.r_last / np.tan(self.hfov)

    def calc_eqfl(self):
        """ 35mm equivalent focal length. For cellphone lens, we usually use EFL to describe the lens.

            35mm sensor: 36mm * 24mm
        """
        return 21.63 / np.tan(self.hfov)

    @torch.no_grad()
    def calc_foc_dist(self, wvln=DEFAULT_WAVE):
        """ Compute the focus distance (object space) of the lens.

            Rays start from sensor and trace to the object space, the focus distance is negative.
        """
        # => Sample point source rays from sensor center
        o1 = torch.tensor([0, 0, self.d_sensor], device=self.device).repeat(GEO_SPP, 1)
        o2 = self.surfaces[0].surface_sample(GEO_SPP)   # A simple method is to sample from the first surface.
        o2 *= 0.2   # Shrink sample region
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=self.device)

        # => Trace rays to the object space and compute focus distance
        ray, _, _ = self.trace(ray)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2) # The solution for the nearest distance.
        focus_p = (ray.o[...,2] - ray.d[...,2] * t)[ray.ra > 0].cpu().numpy()
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p < 0)]
        focus_dist = np.mean(focus_p)

        return focus_dist
    
    @torch.no_grad()
    def refocus_inf(self):
        """ Shift sensor to get the best center focusing.
        """
        # Trace rays and compute in-focus sensor position
        ray = self.sample_parallel_2D(R=self.surfaces[0].r * 0.5, M=GEO_SPP, wvln=DEFAULT_WAVE)
        ray, _, _ = self.trace(ray)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
        focus_p = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
        focus_p = focus_p[ray.ra.cpu() > 0]
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p>0)]
        d_sensor_new = float(np.mean(focus_p))
        
        # Update sensor position
        assert d_sensor_new > 0, 'sensor position is negative.'
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def refocus(self, depth=DEPTH):
        """ Refocus the lens to a depth distance by changing sensor position.

            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Trace green light
        o = self.surfaces[0].surface_sample(GEO_SPP)
        d = o - torch.tensor([0, 0, depth]).to(self.device)
        ray = Ray(o, d, device=self.device)
        ray, _, _ = self.trace(ray)

        # Calculate in-focus sensor position of green light (use least-squares solution)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
        t = t * ray.ra
        focus_d = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
        focus_d = focus_d[ray.ra.cpu() > 0]
        focus_d = focus_d[~np.isnan(focus_d) & (focus_d>0)]
        d_sensor_new = float(np.mean(focus_d))
        
        # Update sensor position
        assert d_sensor_new > 0, 'sensor position is negative.'
        self.d_sensor = d_sensor_new # d_sensor should be a float value, not a tensor

        # FoV will be slightly changed
        self.post_computation()


    # ---------------------------
    # 2. FoV-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_fov(self):
        """ Compute half diagonal fov.

            Shot rays from edge of sensor, trace them to the object space and compute
            angel, output rays should be parallel and the angle is half of fov.
        """
        # Sample rays going out from edge of sensor, shape [M, 3] 
        M = 100
        o1 = torch.zeros([M, 3])
        o1 = torch.tensor([self.r_last, 0, self.d_sensor]).repeat(M, 1)

        pupilz, pupilx = self.exit_pupil()
        x2 = torch.linspace(- pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        ray = self.trace2obj(ray)

        # compute fov
        tan_fov = ray.d[...,0] / ray.d[...,2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))

        if torch.isnan(fov):
            print('computed fov is NaN, use 0.5 rad instead.')
            fov = 0.5
        else:
            fov = fov.item()
        
        return fov

    @torch.no_grad()
    def calc_magnification3(self, depth):
        """ Use mapping relationship (ray tracing) to compute magnification. The computed magnification is very accurate.

            Advatages: can use many data points to reduce error.
            Disadvantages: due to distortion, some data points contain error
        """
        M = GEO_GRID
        spp = 512
        ray = self.sample_point_source(M=M, spp=spp, depth=depth, R=-depth*np.tan(self.hfov)*0.5, pupil=True)
        
        # map r1 from object space to sensor space, ground-truth
        o1 = ray.o.detach()[..., :2]
        o1 = torch.flip(o1, [1, 2])
        
        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)

        # use 1/4 part of regions to compute magnification, also to avoid zero values on the axis
        x1 = o1[0,:,:,0]
        y1 = o1[0,:,:,1]
        x2 = torch.sum(o2[...,0] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)
        y2 = torch.sum(o2[...,1] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)

        mag_x = x1 / x2
        tmp = mag_x[:M//2,:M//2]
        mag = 1 / torch.mean(tmp[~tmp.isnan()]).item()

        if mag == 0:
            scale = - depth * np.tan(self.hfov) / self.r_last
            return 1 / scale

        return mag

    @torch.no_grad()
    def calc_principal(self, wvln=DEFAULT_WAVE):
        """ Compute principal (front and back) planes.
        """
        M = GEO_GRID

        # Backward ray tracing for the first principal point
        ray = self.sample_parallel_2D(R=self.surfaces[0].r * 0.1, M=M, forward=False, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[...,2] - out_ray.d[...,2] * t
        front_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        # Forward ray tracing for the second principal point
        ray = self.sample_parallel_2D(R=self.surfaces[0].r * 0.1, M=M, forward=True, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        back_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        return front_principal, back_principal

    @torch.no_grad()
    def calc_scale_pinhole(self, depth):
        """ Assume the first principle point is at (0, 0, 0), use pinhole camera to calculate the scale factor.
        """
        scale = - depth * np.tan(self.hfov) / self.r_last
        return scale
    
    @torch.no_grad()
    def calc_scale_ray(self, depth):
        """ Use ray tracing to compute scale factor.
        """
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
        """ Compute chief ray. We can use chief ray for fov, magnification.
            Chief ray, a ray goes through center of aperture.
        """
        # sample rays with shape [M, 3]
        M = 1000
        pupilz, pupilx = self.exit_pupil()
        o1 = torch.zeros([M, 3])
        o1[:,0] = pupilx
        o1[:,2] = self.d_sensor
        
        x2 = torch.linspace(-pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2-o1, device=self.device)
        inc_ray = ray.clone()
        ray, _, _ = self.trace(ray, lens_range=list(range(self.aper_idx, len(self.surfaces))))

        center_x = torch.min(torch.abs(ray.o[:,0]))
        center_idx = torch.where(torch.abs(ray.o[:,0])==center_x)
        
        return inc_ray.o[center_idx,:], inc_ray.d[center_idx,:]

    # ---------------------------
    # 3. Pupil-related functions
    # ---------------------------
    @torch.no_grad()
    def exit_pupil(self, shrink_pupil=False):
        """ Sample **forward** rays to compute z coordinate and radius of exit pupil. 
            Exit pupil: ray comes from sensor to object space. 
        """
        return self.entrance_pupil(entrance=False, shrink_pupil=shrink_pupil)


    @torch.no_grad()
    def entrance_pupil(self, M=32, entrance=True, shrink_pupil=False):
        """ We sample **backward** rays, return z coordinate and radius of entrance pupil. 
            Entrance pupil: how many rays can come from object space to sensor. 
        """
        if self.aper_idx is None:
            if entrance:
                return self.surfaces[0].d.item(), self.surfaces[0].r
            else:
                return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # sample M forward rays from edge of aperture to last surface.
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r
        ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(M, 1)

        phi = torch.linspace(-0.5, 0.5, M)
        if entrance:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                -torch.cos(phi)
            ), axis=-1)
        else:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                torch.cos(phi)
            ), axis=-1)

        ray = Ray(ray_o, d, device=self.device)

        # ray tracing
        if entrance:
            lens_range = range(0, self.aper_idx)
            ray,_,_ = self.trace(ray, lens_range=lens_range)
        else:
            lens_range = range(self.aper_idx+1, len(self.surfaces))
            ray,_,_ = self.trace(ray, lens_range=lens_range)

        # compute intersection. o1+d1*t1 = o2+d2*t2
        pupilx = []
        pupilz = []
        for i in range(M):
            for j in range(i+1, M):
                if ray.ra[i] !=0 and ray.ra[j]!=0:
                    d1x, d1z, d2x, d2z = ray.d[i,0], ray.d[i,2], ray.d[j,0], ray.d[j,2]
                    o1x, o1z, o2x, o2z = ray.o[i,0], ray.o[i,2], ray.o[j,0], ray.o[j,2]
                    
                    Adet = - d1x * d2z + d2x * d1z
                    B1 = -d1z*o1x+d1x*o1z
                    B2 = -d2z*o2x+d2x*o2z
                    oz = (- B1 * d2z + B2 * d1z) / Adet
                    ox = (B2 * d1x - B1 * d2x) / Adet
                    
                    pupilx.append(ox.item())
                    pupilz.append(oz.item())
        
        if len(pupilx) == 0:
            avg_pupilx = aper_r
            avg_pupilz = 0
        else:
            avg_pupilx = stats.trim_mean(pupilx, 0.1)
            avg_pupilz = stats.trim_mean(pupilz, 0.1)
            if np.abs(avg_pupilz) < EPSILON:
                avg_pupilz = 0

        if shrink_pupil:
            avg_pupilx *= 0.5

        if avg_pupilx < EPSILON:
            if entrance:
                return self.surfaces[0].d.item(), self.surfaces[0].r
            else:
                return self.surfaces[-1].d.item(), self.surfaces[-1].r
        
        return avg_pupilz, avg_pupilx
    

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
        """ Change aperture radius.
        """
        if aper_r is None:
            if foclen is None:
                foclen = self.calc_efl()
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r
        
        self.fnum = self.foclen / aper_r / 2

    
    def set_target_fov_fnum(self, hfov, fnum, imgh=None):
        """ Set FoV, ImgH and F number, only use this function to assign design targets.

            This function now only works for aperture in the front
        """
        if imgh is not None:
            self.r_last = imgh / 2
        self.hfov = hfov
        self.fnum = fnum
        
        self.foclen = self.calc_efl()
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].r = float(aper_r)


    # ---------------------------
    # 2. Lens operation
    # ---------------------------
    def pertub(self):
        """ Randomly perturb all lens surfaces to simulate manufacturing errors. 

        Including:
            (1) surface position, thickness, curvature, and other coefficients.
            (2) surface rotation, tilt, and decenter.
        
        Called for accurate image simulation, together with sensor noise, vignetting, etc.
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].perturb()

    def double(self):
        """ Use double-precision for the lens group.
        """
        for surf in self.surfaces:
            surf.double()


    # ---------------------------
    # 3. Lens pruning
    # ---------------------------
    @torch.no_grad()
    def prune_surf(self, outer=None, surface_range=None):
        """ Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            outer (float): extra height to reserve. 
                For cellphone lens, we usually use 0.1mm or 0.05 * r_last. 
                For camera lens, we usually use 0.5mm or 0.1 * r_last.
        """
        outer = self.r_last * 0.05 if outer is None else outer
        surface_range = self.find_diff_surf() if surface_range is None else surface_range

        # ==> 1. Reset lens to maximum height(sensor radius)
        for i in surface_range:
            self.surfaces[i].r = self.r_last

        # ==> 2. Prune to reserve valid surface height
        # sample maximum fov rays to compute valid surface height
        view = self.hfov if self.hfov is not None else np.arctan(self.r_last/self.d_sensor)
        ray = self.sample_parallel_2D(view=np.rad2deg(view), M=21, entrance_pupil=True)

        ps, oss = self.trace2sensor(ray=ray, record=True)
        for i in surface_range:
            height = []
            for os in oss:  # iterate all rays
                try:
                    # because oss records the starting point at position 0, we need to ignore this.
                    height.append(np.abs(os[i+1][0]))   # the second index 0 means x coordinate
                except:
                    continue

            try:
                self.surfaces[i].r = max(height) + outer
            except:
                continue
        
        # ==> 3. Front surface should be smaller than back surface. This does not apply to fisheye lens.
        for i in surface_range[:-1]:
            if self.materials[i].A < self.materials[i+1].A:
                self.surfaces[i].r = min(self.surfaces[i].r, self.surfaces[i+1].r)

        # ==> 4. Remove nan part, also the maximum height should not exceed sensor radius
        for i in surface_range:
            max_height = min(self.surfaces[i].max_height(), self.r_last)
            self.surfaces[i].r = min(self.surfaces[i].r, max_height)


    @torch.no_grad()
    def correct_shape(self):
        """ Correct wrong lens shape during the lens design.
        """
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
            d_aper = 0.1

            # If the first surface is concave, use the maximum negative sag. 
            aper_r = self.surfaces[aper_idx].r
            sag1 = - self.surfaces[aper_idx+1].surface(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in diff_surf_range:
                self.surfaces[i].d -= delta_aper

        
        # ==> Rule 3: If two surfaces overlap (at center), seperate them by a small distance
        for i in diff_surf_range[:-1]:
            if self.surfaces[i].d > self.surfaces[i+1].d:
                self.surfaces[i+1].d += 0.2
                shape_changed = True

        # ==> Rule 4: Prune all surfaces
        self.prune_surf()

        if shape_changed:
            print('Surface shape corrected.')
        return shape_changed


    # ====================================================================================
    # Visualization.
    # ====================================================================================
    @torch.no_grad()
    def analysis(self, save_name='./test', render=False, multi_plot=False, plot_invalid=True, zmx_format=True, depth=DEPTH, render_unwarp=False, lens_title=None):
        """ Analyze the optical lens.
        """
        # Draw lens geometry and ray path
        self.plot_setup2D_with_trace(filename=save_name, multi_plot=multi_plot, entrance_pupil=True, plot_invalid=plot_invalid, zmx_format=zmx_format, lens_title=lens_title, depth=depth)

        # Draw spot diagram and PSF map
        self.draw_psf_map(save_name=save_name, ks=61)

        # Calculate RMS error
        rms_avg, rms_radius_on_axis, rms_radius_off_axis = self.analysis_rms()
        print(f'On-axis RMS radius: {round(rms_radius_on_axis.item()*1000,3)}um, Off-axis RMS radius: {round(rms_radius_off_axis.item()*1000,3)}um, Avg RMS spot size (radius): {round(rms_avg.item()*1000,3)}um.')

        # Render an image, compute PSNR and SSIM
        if render:
            img_org = cv.cvtColor(cv.imread(f'./datasets/usaf1951.png'), cv.COLOR_BGR2RGB)
            img_render = self.render_single_img(img_org, depth=depth, spp=128, unwarp=render_unwarp, save_name=f'{save_name}_render', noise=0.01)

            render_psnr = round(compare_psnr(img_org, img_render, data_range=255), 4)
            render_ssim = round(compare_ssim(img_org, img_render, channel_axis=2, data_range=255), 4)
            print(f'Rendered image: PSNR={render_psnr}, SSIM={render_ssim}')


    @torch.no_grad()      
    def plot_setup2D_with_trace(self, filename, views=[0], M=7, depth=None, entrance_pupil=True, zmx_format=False, plot_invalid=True, multi_plot=False, lens_title=None):
        """ Plot lens setup with rays.
        """
        # ==> Title
        if lens_title is None:
            if self.aper_idx is not None:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_F/{round(self.fnum,2)}_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
            else:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
        

        # ==> Plot RGB seperately
        if multi_plot:
            R = self.surfaces[0].r
            views = np.linspace(0, np.rad2deg(self.hfov)*0.99, num=7)
            colors_list = 'rgb'
            fig, axs = plt.subplots(1, 3, figsize=(24, 5))
            fig.suptitle(lens_title)

            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.plot_setup2D(ax=ax, fig=fig, zmx_format=zmx_format)

                for view in views:
                    if depth is None:
                        ray = self.sample_parallel_2D(R, wvln, view=view, M=M, entrance_pupil=entrance_pupil)
                    else:
                        ray = self.sample_point_source_2D(depth=depth, view=view, M=M, entrance_pupil=entrance_pupil, wvln=wvln)
                    ps, oss = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)
                    ax.axis('off')

            fig.savefig(f"{filename}.svg", bbox_inches='tight', format='svg', dpi=600)
            fig.savefig(f"{filename}.png", bbox_inches='tight', format='png', dpi=300)
            plt.close()
        

        # ==> Plot RGB in one figure
        else:
            R = self.surfaces[0].r
            colors_list = 'bgr'
            views = [0, np.rad2deg(self.hfov)*0.707, np.rad2deg(self.hfov)*0.99]
            aspect = self.sensor_res[1] / self.sensor_res[0]
            ax, fig = self.plot_setup2D(zmx_format=zmx_format)
            
            for i, view in enumerate(views):
                if depth is None:
                    ray = self.sample_parallel_2D(R, WAVE_RGB[2-i], view=view, M=M, entrance_pupil=entrance_pupil)
                else:
                    ray = self.sample_point_source_2D(depth=depth, view=view, M=M, entrance_pupil=entrance_pupil, wvln=WAVE_RGB[2-i])
                        
                ps, oss = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)

            ax.axis('off')
            ax.set_title(lens_title, fontsize=10)
            fig.savefig(f"{filename}.png", format='png', dpi=600)
            plt.close()

    
    def plot_back_ray_trace(self, filename='debug_backward_rays', spp=5, vpp=5, pupil=True):
        ax, fig = self.plot_setup2D()

        ray = self.sample_sensor_2D(pupil=pupil, spp=spp, vpp=vpp)
        _, _, oss = self.trace(ray=ray, record=True)
        ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color='b')

        ax.axis('off')
        fig.savefig(f"{filename}.png", bbox_inches='tight')


    def plot_raytraces(self, oss, ax=None, fig=None, color='b-', show=True, p=None, valid_p=None, plot_invalid=True, ra=None):
        """ Plot ray paths.
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D()
        else:
            show = False

        for i, os in enumerate(oss):
            o = torch.Tensor(np.array(os)).to(self.device)
            x = o[...,0]
            z = o[...,2]

            o = o.cpu().detach().numpy()
            z = o[...,2].flatten()
            x = o[...,0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i,0])
                    z = np.append(z, p[i,2])

            if plot_invalid:
                ax.plot(z, x, color, linewidth=0.8)
            elif ra[i]>0:
                ax.plot(z, x, color, linewidth=0.8)

        if show: 
            plt.show()
        else: 
            plt.close()

        return ax, fig


    def plot_setup2D(self, ax=None, fig=None, color='k', with_sensor=True, zmx_format=False, fix_bound=False):
        """ Draw lens setup.
        """
        def plot(ax, z, x, color):
            p = torch.stack((x, torch.zeros_like(x, device=self.device), z), axis=-1)
            p = p.cpu().detach().numpy()
            ax.plot(p[...,2], p[...,0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d.item() - APERTURE_WEDGE_LENGTH, d.item() + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)        

        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            show=False

        if len(self.surfaces) == 1: # if there is only one surface, then it should be aperture
            # draw_aperture(ax, self.surfaces[0], color='orange')
            raise Exception('Only one surface is not supported.')

        else:
            # Draw lens
            for i, s in enumerate(self.surfaces):

                # Draw aperture
                if isinstance(s, Aperture):
                    draw_aperture(ax, s, color='orange')

                # Draw lens surface
                else:
                    r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                    z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))   # graw surface
                    plot(ax, z, r, color)

            # Connect two surfaces
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if s.mat1.A < 1.0003: # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)

                    if zmx_format:
                        if r > r_prev:
                            z = torch.stack((sag_prev, sag_prev, sag))
                            x = torch.Tensor(np.array([[r_prev], [r], [r]])).to(self.device)
                        else:
                            z = torch.stack((sag_prev, sag, sag))
                            x = torch.Tensor(np.array([[r_prev], [r_prev], [r]])).to(self.device)
                    else:
                        z = torch.stack((sag_prev, sag))
                        x = torch.Tensor(np.array([[r_prev], [r]])).to(self.device)

                    plot(ax, z, x, color)
                    plot(ax, z,-x, color)
                    s_prev = s

            # Draw sensor
            if with_sensor:
                ax.plot([self.d_sensor, self.d_sensor], [-self.r_last, self.r_last], color)
        
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        
        if fix_bound:
            ax.set_aspect('equal')
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect('equal', adjustable='datalim', anchor='C') 
            ax.minorticks_on() 
            ax.set_xlim(-0.5, 7.5) 
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig


    @torch.no_grad()
    def draw_psf_map(self, grid=7, depth=DEPTH, ks=51, log_scale=False, quater=False, save_name=None):
        """ Draw RGB PSF map at a certain depth. Will draw M x M PSFs, each of size ks x ks.
        """
        # Calculate PSF map
        psf_map = self.psf_map(depth=depth, grid=grid, ks=ks, spp=GEO_SPP, center=True)
        
        # Normalize for each field
        for i in range(0, psf_map.shape[-2], ks):
            for j in range(0, psf_map.shape[-1], ks):
                psf_map[:,i:i+ks,j:j+ks] /= psf_map[:,i:i+ks,j:j+ks].max()

        # Los scale the PSF for better visualization
        if log_scale:
            psf_map = torch.log(psf_map + 1e-3)   # 1e-3 is an empirical value

        # Save figure using matplotlib
        plt.figure(figsize=(10, 10))
        psf_map = psf_map.permute(1, 2, 0).cpu().numpy()
        plt.imshow(psf_map)

        H, W = psf_map.shape[:2]
        ruler_len = 100
        arrow_end = ruler_len / (self.pixel_size * 1e3)   # plot a scale ruler
        plt.annotate('', xy=(0, H - 10), xytext=(arrow_end, H - 10), arrowprops=dict(arrowstyle='<->', color='white'))
        plt.text(arrow_end + 10, H - 10, f'{ruler_len} um', color='white', fontsize=12, ha='left')
        
        plt.axis('off')
        plt.tight_layout(pad=0)  # Removes padding
        save_name = f'./psf{-depth}mm.png' if save_name is None else f'{save_name}_psf{-depth}mm.png'
        plt.savefig(save_name, dpi=300)
        plt.close()


    @torch.no_grad()
    def draw_psf_radial(self, M=3, depth=DEPTH, ks=51, log_scale=False, save_name='./psf_radial.png'):
        """ Draw radial PSF (45 deg). Will draw M PSFs, each of size ks x ks.  
        """
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
        """ Draw spot diagram of the lens. Shot rays from grid points in object space, trace to sensor and visualize.
        """
        # Sample and trace rays from grid points
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, wvln=wvln, spp=1024, pupil=True)
        ray = self.trace2sensor(ray)
        o2 = - ray.o.clone().cpu().numpy()
        ra = ray.ra.clone().cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(M, M, figsize=(30,30))
        for i in range(M):
            for j in range(M):
                ra_ = ra[:,i,j]
                x, y = o2[:,i,j,0], o2[:,i,j,1]
                x, y = x[ra_>0], y[ra_>0]
                xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

                # scatter plot
                axs[i, j].scatter(x, y, 1, 'black')
                axs[i, j].scatter([xc], [yc], None, 'r', 'x')
                axs[i, j].set_aspect('equal', adjustable='datalim')
        
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)

        plt.close()


    @torch.no_grad()
    def draw_spot_radial(self, M=3, depth=DEPTH, save_name=None):
        """ Draw radial spot diagram of the lens.

        Args:
            M (int, optional): field number. Defaults to 3.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample and trace rays
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(M=M*2-1, R=self.sensor_size[0]/2/mag, depth=depth, spp=1024, pupil=True, wvln=589.3)
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        o2 = torch.flip(ray.o.clone(), [1, 2]).cpu().numpy()
        ra = torch.flip(ray.ra.clone(), [1, 2]).cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(1, M, figsize=(M*12,10))
        for i in range(M):
            i_bias = i + M - 1

            # calculate center of mass
            ra_ = ra[:,i_bias,i_bias]
            x, y = o2[:,i_bias,i_bias,0], o2[:,i_bias,i_bias,1]
            x, y = x[ra_>0], y[ra_>0]
            xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

            # scatter plot
            axs[i].scatter(x, y, 12, 'black')
            axs[i].scatter([xc], [yc], 400, 'r', 'x')
            
            # visualization
            axs[i].set_aspect('equal', adjustable='datalim')
            axs[i].tick_params(axis='both', which='major', labelsize=18)
            axs[i].spines['top'].set_linewidth(4)
            axs[i].spines['bottom'].set_linewidth(4)
            axs[i].spines['left'].set_linewidth(4)
            axs[i].spines['right'].set_linewidth(4)

        # Save figure
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)

        plt.close()


    @torch.no_grad()
    def draw_mtf(self, relative_fov=[0.0, 0.7, 1.0], save_name='./mtf.png', wvlns=DEFAULT_WAVE, depth=DEPTH):
        """ Draw MTF curve of the lens. 
        """
        if save_name[-4:] != '.png':
            save_name += '.png'

        relative_fov = [relative_fov] if isinstance(relative_fov, float) else relative_fov
        wvlns = [wvlns] if isinstance(wvlns, float) else wvlns
        color_list = 'rgb'

        plt.figure(figsize=(6,6))
        for wvln_idx, wvln in enumerate(wvlns):
            for fov_idx, fov in enumerate(relative_fov):
                point = torch.Tensor([fov, fov, depth])
                psf = self.psf(points=point, wvln=wvln, ks=256)
                freq, mtf_tan, mtf_sag = self.psf2mtf(psf)

                fov_deg = round(fov * self.hfov * 57.3, 1)
                plt.plot(freq, mtf_tan, color_list[fov_idx], label=f'{fov_deg}(deg)-Tangential')
                plt.plot(freq, mtf_sag, color_list[fov_idx], label=f'{fov_deg}(deg)-Sagittal', linestyle='--')

        plt.legend()
        plt.xlabel('Spatial Frequency [cycles/mm]')
        plt.ylabel('MTF')

        # Save figure
        plt.savefig(f'{save_name}', bbox_inches='tight', format='png', dpi=300)
        plt.close()

    def draw_distortion(self, depth=DEPTH, save_name=None):
        """ Draw distortion.
        """
        # Ray tracing to calculate distortion map
        M = 15
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(M=M, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True)
        o1 = ray.o.detach().cpu()
        x1 = o1[0,:,:,0] / scale 
        y1 = o1[0,:,:,1] / scale 

        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o2 = o2.clone().cpu()
        x2 = torch.sum(o2[...,0] * ray.ra.cpu(), axis=0)/ torch.sum(ray.ra.cpu(), axis=0)
        y2 = torch.sum(o2[...,1] * ray.ra.cpu(), axis=0)/ torch.sum(ray.ra.cpu(), axis=0)

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title('Lens distortion')
        ax.scatter(x1, y1, s=2)
        ax.scatter(x2, y2, s=2)
        ax.legend(['ref', 'distortion'])
        ax.axis('scaled')

        if save_name is None:
            plt.savefig(f'./distortion{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)
        else:
            plt.savefig(f'{save_name}_distortion{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)


    # ====================================================================================
    # Distortion
    # ====================================================================================
    @torch.no_grad()
    def unwarp(self, img, depth, grid=256, spp=256, crop=True):
        """ Unwarp rendered images.
        """
        # Ray tracing to calculate distortion map
        scale = self.calc_scale_ray(depth)
        ray = self.sample_point_source(M=grid, spp=spp, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True)
        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o_dist = (o2*ray.ra.unsqueeze(-1)).sum(0)/ray.ra.sum(0).add(EPSILON).unsqueeze(-1)   # shape (H, W, 2)

        # Reshape to [N, C, H, W], normalize to [-1, 1], then resize to img resolution [N, C, H, W]
        x_dist = F.interpolate(-o_dist.unsqueeze(0).unsqueeze(0)[..., 0] / self.sensor_size[1] * 2, img.shape[-2:], mode='bilinear', align_corners=True)
        y_dist = F.interpolate(o_dist.unsqueeze(0).unsqueeze(0)[..., 1] / self.sensor_size[0] * 2, img.shape[-2:], mode='bilinear', align_corners=True)
        grid_dist = torch.stack((x_dist.squeeze(0), y_dist.squeeze(0)), dim=-1)
        
        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(img, grid_dist, align_corners=True)
        
        return img_unwarpped

    # ====================================================================================
    # Loss function
    # ====================================================================================
    def loss_infocus(self, bound=0.005):
        """ Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

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
            rms_size = torch.sqrt(torch.sum(p**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra))
            loss.append(max(rms_size, bound))
        
        loss_avg = sum(loss) / len(loss)
        return loss_avg


    def analysis_rms(self, depth=DEPTH, ref=True):
        """ Compute RMS-based error. Contain both RMS errors and RMS radius.
        """
        grid = 10
        x = torch.linspace(0, 1, grid)
        y = torch.linspace(0, 1, grid)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)
        scale = self.calc_scale_ray(depth)

        # Ray position in the object space by perspective projection, because points are normalized
        point_obj_x = points[..., 0] * scale * self.sensor_size[1] / 2   # x coordinate
        point_obj_y = points[..., 1] * scale * self.sensor_size[0] / 2   # y coordinate
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1) 
        
        # Point center determined by green light
        ray = self.sample_from_points(o=point_obj, spp=GEO_SPP, wvln=DEFAULT_WAVE)
        ray = self.trace2sensor(ray)
        pointc_green = (ray.o[..., :2] * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(0.0001).unsqueeze(-1)

        # Calculate RMS spot size 
        rms = []
        for wvln in WAVE_RGB:
            # Trace rays to sensor plane
            ray = self.sample_from_points(o=point_obj, spp=GEO_SPP, wvln=wvln)
            ray = self.trace2sensor(ray)

            # Calculate RMS error for different FoVs
            o2_norm = (ray.o[..., :2] - pointc_green) * ray.ra.unsqueeze(-1)
            rms0 = torch.sqrt((o2_norm**2 * ray.ra.unsqueeze(-1)).sum((0, 2)) / ray.ra.sum(0))
            rms.append(rms0)
            
        rms = torch.stack(rms, dim=0)
        rms = torch.mean(rms, dim=0)

        # Calculate RMS error for on-axis and off-axis
        rms_avg = rms.mean()
        rms_radius_on_axis = rms[0]
        rms_radius_off_axis = rms[-1]

        return rms_avg, rms_radius_on_axis, rms_radius_off_axis
    

    def loss_rms(self, depth=DEPTH, show=False):
        """ Compute RGB RMS error per pixel, forward rms error.

            Can also revise this function to plot PSF.
        """
        H = 31

        # ==> PSF and RMS by patch
        scale = - depth * np.tan(self.hfov) / self.r_last

        rms = 0
        for wvln in WAVE_RGB:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wvln=wvln)
            ray, _, _ = self.trace(ray)
            o2 = ray.project_to(self.d_sensor)
            o2_center = (o2*ray.ra.unsqueeze(-1)).sum(0)/ray.ra.sum(0).add(0.0001).unsqueeze(-1)    
            o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)   # normalized to center (0, 0)
            rms += torch.sum(o2_norm**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra)

        return rms / 3


    def loss_surface(self, grad_bound=0.5):
        """ Surface should be smooth, aggressive shape change should be pealized. 
        """
        loss = 0.
        for i in self.find_diff_surf():
            r = self.surfaces[i].r
            loss += max(self.surfaces[i]._dgd(r**2).abs(), grad_bound)

        return loss


    def loss_self_intersec(self, dist_bound=0.1, thickness_bound=0.4):
        """ Loss function to avoid self-intersection. Loss is designed by the distance to the next surfaces.
        
        Args:
            dist_bound (float): distance bound.
            thickness_bound (float): thickness bound.

        Parameter settings:
            For cellphone lens: dist_bound=0.1, thickness_bound=0.4
            For camera lens: dist_bound=1.0, thickness_bound=1.0
            General: dist_bound=thickness_bound=0.5 * (total_thickness_of_lens / lens_element_number / 2)
        """
        loss = 0.
        diff_surf = self.find_diff_surf()
        for i in range(len(diff_surf) - 1):
            current_surf = self.surfaces[diff_surf[i]]
            next_surf = self.surfaces[diff_surf[i+1]]

            r = torch.tensor([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).to(self.device) * current_surf.r
            z_front = current_surf.surface(r, 0) + current_surf.d
            z_next = next_surf.surface(r, 0) + next_surf.d
            dist_min = torch.min(z_next - z_front)

            if self.materials[diff_surf[i]].name == 'air':
                loss += min(thickness_bound, dist_min)
            else:
                loss += min(dist_bound, dist_min)

        return -loss


    def loss_last_surf(self, dist_bound=0.6):
        """ The last surface should not hit the sensor plane.

            There should also be space for IR filter.
        """
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.6, 1, 11).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface(r, 0) - last_surf.d
        loss = min(dist_bound, torch.min(z_last_surf))
        return - loss
 

    def loss_ray_angle(self, target=0.7, depth=DEPTH):
        """ Loss function designed to penalize large incident angle rays.

            Reference value: > 0.7
        """
        # Sample rays [spp, M, M]
        M = 11
        spp = 32
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(M=M, spp=spp, depth=DEPTH, R=scale*self.sensor_size[0]/2, pupil=True)

        # Ray tracing
        ray, _, _ = self.trace(ray)

        # Loss (we want to maximize ray angle term)
        loss = torch.sum(ray.obliq * ray.ra) / (torch.sum(ray.ra) + EPSILON)
        loss = torch.min(loss, torch.Tensor([target]).to(self.device))

        return - loss


    def loss_reg(self):
        """ An empirical regularization loss for lens design.
        """
        if self.is_cellphone:
            loss_reg = 0.1 * self.loss_infocus() + self.loss_ray_angle() + (self.loss_self_intersec() + self.loss_last_surf())
        else:
            loss_reg = 0.1 * self.loss_infocus() + self.loss_self_intersec(dist_bound=0.5, thickness_bound=0.5)
        
        return loss_reg
    


    # ====================================================================================
    # Optimization
    # ====================================================================================
    def activate_surf(self, activate=True, diff_surf_range=None):
        """ Activate gradient for each surface.
        """
        if diff_surf_range is None:
            diff_surf_range = range(len(self.surfaces))
            if self.aper_idx is not None:
                del diff_surf_range[self.aper_idx]

        for i in diff_surf_range:
            self.surfaces[i].activate_grad(activate)


    def get_optimizer_params(self, lr=[1e-4, 1e-4, 0, 1e-4], decay=0.2, diff_surf_range=None):
        """ Get optimizer parameters for different lens surface.

            For cellphone lens: [c, d, k, a], [1e-4, 1e-4, 1e-1, 1e-4]
            For camera lens: [c, d, 0, 0], [1e-3, 1e-4, 0, 0]

        Args:
            lr (list): learning rate for different parameters [c, d, k, a]. Defaults to [1e-4, 1e-4, 0, 1e-4].
            decay (float): decay rate for higher order a. Defaults to 0.2.
            diff_surf_range (list): surface indices to be optimized. Defaults to None.

        Returns:
            list: optimizer parameters
        """
        diff_surf_range = self.diff_surf_range if diff_surf_range is None else diff_surf_range
        params = []
        for i in diff_surf_range:
            surf = self.surfaces[i]
            
            if isinstance(surf, Aperture):
                pass
            
            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(lr=lr, decay=decay)
            
            # elif isinstance(surf, DOE_GEO):
            #     params += surf.get_optimizer_params(lr=lr[0])
            
            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(lr=lr[:2])

            else:
                raise Exception('Surface type not supported yet.')

        return params

    def get_optimizer(self, lr=[1e-4, 1e-4, 0, 1e-4], decay=0.2):
        """ Get optimizers and schedulers for different lens parameters.

        Args:
            lrs (_type_): _description_
            epochs (int, optional): _description_. Defaults to 100.
            ai_decay (float, optional): _description_. Defaults to 0.2.
        """
        params = self.get_optimizer_params(lr, decay)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def refine(self, lrs=[5e-4, 1e-4, 0.1, 1e-4], decay=0.1, iterations=2000, test_per_iter=100, depth=DEPTH, shape_control=True, centroid=False, importance_sampling=False, result_dir='./results'):
        """ Optimize/Refine a given lens by minimizing rms errors.
            
            L1: small deviation between s and t 
            L2: large deviation between s and t

            refocus: center better
            no-refocus: more balanced

        Args:
            depth (float, optional): Depth of scene images. Defaults to DEPTH.
            lrs (list): Learning rate list. Lr for [c, d, k, ai]. Defaults to [1e-3, 1e-4, 0.1, 1e-4]. Ref: from scratch 1e-2, fine-tuning 1e-4
            decay (float, optional): Learning rate alpha decay. Defaults to 0.1.
        """
        # Preparation
        num_grid = 21
        spp = 512
        w_reg = 0.02
        sample_rays_per_iter = test_per_iter if not centroid else 5 * test_per_iter
        optimizer = self.get_optimizer(lrs, decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=iterations//10, num_training_steps=iterations)

        result_dir = result_dir + '/' + datetime.now().strftime("%m%d-%H%M%S")+ '-DesignLensRMS'
        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(f'lr:{lrs}, decay:{decay}, iterations:{iterations}, spp:{spp}, grid:{num_grid}.')

        # Training
        pbar = tqdm(total=iterations+1, desc='Progress', postfix={'rms': 0})
        for i in range(iterations+1):
            # =========================================
            # Evaluate the lens
            # =========================================
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if i > 0 and shape_control:   
                        self.correct_shape()

                    self.write_lens_json(f'{result_dir}/iter{i}.json')
                    self.analysis(f'{result_dir}/iter{i}', zmx_format=True, plot_invalid=True, multi_plot=False)
                    
            # =========================================
            # Compute centriod and sample new rays
            # =========================================
            if i % sample_rays_per_iter == 0:
                # ==> Use center spot of green rays, no gradient computation
                with torch.no_grad():
                    mag = 1 / self.calc_scale_pinhole(depth)
                    ray = self.sample_point_source(M=num_grid, R=self.sensor_size[0]/2/mag, depth=depth, spp=spp*4, pupil=True, wvln=DEFAULT_WAVE, importance_sampling=importance_sampling)
                    xy_center_ref = - ray.o[0, :, :, :2] * mag

                    ray, _, _ = self.trace(ray)
                    ray.propagate_to(self.d_sensor)
                    xy_center = (ray.o[...,:2]*ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(EPSILON).unsqueeze(-1)
                    
                    # center_p shape [M, M, 2]
                    if centroid:
                        center_p = xy_center
                    else:
                        center_p = xy_center_ref
            
                # ==> Sample new rays for training
                rays_backup = []
                for wv in WAVE_RGB:
                    ray = self.sample_point_source(M=num_grid, R=self.sensor_size[0]/2/mag, depth=depth, spp=spp, pupil=True, wvln=wv, importance_sampling=importance_sampling)
                    rays_backup.append(ray)
                   

            # =========================================
            # Optimize lens by minimizing rms
            # =========================================
            loss_rms = []
            for j, wv in enumerate(WAVE_RGB):
                # => Ray tracing
                ray = rays_backup[j].clone()
                ray, _, _ = self.trace(ray)
                xy = ray.project_to(self.d_sensor)
                xy_norm = (xy - center_p) * ray.ra.unsqueeze(-1)

                # => Masked RMS loss function
                weight_mask = torch.sqrt((xy_norm.clone().detach()**2).sum([0, -1]) / (ray.ra.sum([0]) + EPSILON)) # Use L2 error as weight mask
                weight_mask /= weight_mask.mean()
                weight_mask[weight_mask < 0.9] *= 0.1    # Drop out well-trained regions. Very helpful! When a lens is well-trained, we prefer not dropping out
                l_rms = torch.sum(xy_norm.abs().sum(-1) * weight_mask) / (torch.sum(ray.ra) + EPSILON) # weighted L1 loss
                loss_rms.append(l_rms)

            loss_rms = sum(loss_rms) / len(loss_rms)

            # => Regularization
            loss_reg = self.loss_reg()
            L_total = loss_rms + w_reg * loss_reg

            # => Gradient-based optimization
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(rms=loss_rms.item())
            pbar.update(1)

        # Finish training
        pbar.close()


    # ====================================================================================
    # Lesn file IO
    # ====================================================================================
    def write_lens_json(self, filename='./test.json'):
        """ Write the lens into .json file.
        """
        data = {}
        data['foclen'] = self.foclen
        data['fnum'] = self.fnum
        data['r_last'] = self.r_last
        data['d_sensor'] = self.d_sensor
        data['sensor_size'] = self.sensor_size
        data['surfaces'] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = s.surf_dict()
            
            if i < len(self.surfaces) - 1:
                surf_dict['d_next'] = self.surfaces[i+1].d.item() - self.surfaces[i].d.item()
            else:
                surf_dict['d_next'] = self.d_sensor - self.surfaces[i].d.item()
            
            data['surfaces'].append(surf_dict)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def read_lens_json(self, filename='./test.json'):
        """ Read the lens from .json file.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, 'r') as f:
            data = json.load(f)
            d = 0.0
            for surf_dict in data['surfaces']:

                if surf_dict['type'] == 'Aperture':
                    s = Aperture(r=surf_dict['r'], d=d)
                
                elif surf_dict['type'] == 'Aspheric':
                    s = Aspheric(c=surf_dict['c'], r=surf_dict['r'], d=d, k=surf_dict['k'], ai=surf_dict['ai'], mat1=surf_dict['mat1'], mat2=surf_dict['mat2'])

                elif surf_dict['type'] == 'Cubic':
                    s = Cubic(r=surf_dict['r'], d=d, b=surf_dict['b'], mat1=surf_dict['mat1'], mat2=surf_dict['mat2'])

                # elif surf_dict['type'] == 'DOE_GEO':
                #     s = DOE_GEO(l=surf_dict['l'], d=d, glass=surf_dict['glass'])

                elif surf_dict['type'] == 'Plane':
                    s = Plane(l=surf_dict['l'], d=d, mat1=surf_dict['mat1'], mat2=surf_dict['mat2'])
                
                elif surf_dict['type'] == 'Stop':
                    s = Aperture(r=surf_dict['r'], d=d)

                elif surf_dict['type'] == 'Spheric':
                    c = 1/surf_dict['roc'] if surf_dict['roc'] != 0 else 0
                    s = Spheric(c=c, r=surf_dict['r'], d=d, mat1=surf_dict['mat1'], mat2=surf_dict['mat2'])
                    
                else:
                    raise Exception('Surface type not implemented.')
                
                self.surfaces.append(s)
                d += surf_dict['d_next']

        self.sensor_size = data['sensor_size']
        self.d_sensor = d


    def write_zmx(self, filename='./test.zmx'):
        """ Write the lens into .zmx file.
        """
        lens_zmx_str = ''
        
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
            d_next = self.surfaces[i+1].d - self.surfaces[i].d if i < len(self.surfaces)-1 else self.d_sensor - self.surfaces[i].d
            surf_str = s.zmx_str(surf_idx=i+1, d_next=d_next)
            lens_zmx_str += surf_str
        
        # Sensor string
        sensor_str = f"""SURF {i+2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_last}
"""
        lens_zmx_str += sensor_str

        # Write lens zmx string into file
        with open(filename, 'w') as f:
            f.writelines(lens_zmx_str)
            f.close()


# ====================================================================================
# Other functions.
# ====================================================================================
def create_lens(rff=1.0, flange=1.0, d_aper=0.5, d_sensor=None, hfov=0.6, imgh=6, fnum=2.8, surfnum=4, dir='.'):
    """ Create a flat starting point for cellphone lens design.

        Aperture is placed 0.2mm i front of the first surface.

    Args:
        r: relative flatness factor. r = L/img_r(diag)
        flange: distance from last surface to sensor
        d_sensor: total distance of then whole system. usually as a constraint in designing cellphone lens.
        fov: half diagonal fov in radian.
        foclen: focus length.
        fnum: maximum f number.
        surfnum: number of pieces to use.
    """ 
    # ==> Calculate parameters
    foclen = imgh / 2 / np.tan(hfov)
    aper = foclen / fnum / 2
    if d_sensor is None:
        d_sensor = imgh * rff # total thickness

    d_opt = d_sensor - flange - d_aper
    partition = np.clip(np.random.randn(2*surfnum-1) + 1, 1, 2) # consider aperture, position normalization
    partition = [p if i%2==0 else (0.6+0.05*i)*p for i,p in enumerate(partition)]    # distance between lenses should be small
    partition[-2] *= 1.2  # the last lens should be far from the last second one
    partition = partition/np.sum(partition)
    d_ls = partition * d_opt
    d_ls = np.insert(d_ls, 0, 0)
    d_total = 0
    
    mat_table = MATERIAL_TABLE
    mat_names = ['coc', 'okp4', 'pmma', 'pc', 'ps']
    
    # ==> Add surfaces and materials
    surfaces = []
    materials = []

    # 1. add aperture in the front
    surfaces.append(Aspheric(aper, 0))
    materials.append(Material(name='occluder'))
    d_total += d_aper
    
    # 2. add lens surface sequentially
    for i in range(2*surfnum):
        # surface
        d_total += d_ls[i]
        surfaces.append(Aspheric(imgh/2, d_total)) 
        
        # material (last meterial)
        n = 1
        if i % 2 != 0:
            while(n==1):    # randomly select a new glass until n!=1
                mat_name = random.choice(mat_names)
                n, v = mat_table[mat_name]
        mat = Material(name='air') if n==1 else Material(name=mat_name) 
        materials.append(mat)
    
    # 3. add sensor
    materials.append(Material(name='air'))
    
    lens = Lensgroup()
    lens.load_external(surfaces, materials, imgh/2, d_sensor)
    lens.write_lens_json(f'{dir}/starting_point_hfov{hfov}_imgh{imgh}_fnum{fnum}.json')
    return lens


def read_zmx(filename='./test.zmx'):
        """ Read .zmx file into lens file in our format.

        Args:
            filename (str, optional): .zmx file. Defaults to './test.zmx'.
        """
        assert filename[-4:] == '.zmx'
        with open(filename, 'r') as f:
            content = f.read()

        write_surf = False
        total_d = 0
        surfaces = []
        materials = []
        surf_mate = 'AIR'
        mate = Material(name=surf_mate)
        materials.append(mate)
        surf_k = 0.0
        
        lines = content.split('\n')
        for line in lines:
            words = list(filter(None, line.split(' '))) # remove ' '
            if len(words) == 0:
                continue
            
            if words[0] == 'SURF':
                surf_info = True
                surf_idx = int(words[1])

                # append last surface
                if surf_idx == 0 or surf_idx == 1:
                    continue
                else:
                    if surf_type == 'STANDARD':
                        surf = Aspheric(r=surf_r, d=total_d-surf_d)
                        surfaces.append(surf)
                        mate = Material(name=surf_mate)
                        materials.append(mate)
                    elif surf_type == 'EVENASPH':
                        surf_ai = [surf_a2, surf_a4, surf_a6, surf_a8, surf_a10, surf_a12, surf_a14, surf_a16]
                        surf_ai = [ai for ai in surf_ai if ai!=0]
                        surf = Aspheric(r=surf_r, d=total_d-surf_d, c=surf_c, k=surf_k, ai=surf_ai)
                        surfaces.append(surf)
                        mate = Material(name=surf_mate)
                        materials.append(mate)
                        surf_mate = 'AIR'
                    
                    continue

            elif words[0] == 'STOP':
                aper = True

            elif words[0] == 'TYPE':
                surf_type = words[1]
            
            elif words[0] == 'CURV':
                surf_c = float(words[1])

            elif words[0] == 'PARM':
                if words[1] == '1':
                    surf_a2 = float(words[2])
                elif words[1] == '2':
                    surf_a4 = float(words[2])
                elif words[1] == '3':
                    surf_a6 = float(words[2])
                elif words[1] == '4':
                    surf_a8 = float(words[2])
                elif words[1] == '5':
                    surf_a10 = float(words[2])
                elif words[1] == '6':
                    surf_a12 = float(words[2])
                elif words[1] == '7':
                    surf_a14 = float(words[2])
                elif words[1] == '8':
                    surf_a16 = float(words[2])

            elif words[0] == 'DISZ':
                surf_d = float(words[1])
                if surf_d != float('inf'):
                    total_d += surf_d

            elif words[0] == 'CONI':
                surf_k = float(words[1])

            elif words[0] == 'DIAM':
                surf_r = float(words[1])

            elif words[0] == 'GLAS':
                surf_mate = words[1]

        # Sensor
        lens = Lensgroup()
        lens.load_external(surfaces, materials, surf_r, total_d)
        lens.write_lens_json(f'{filename[:-4]}.json')

        return lens

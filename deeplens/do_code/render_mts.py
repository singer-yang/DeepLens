""" Render Mitsuba2 scene by Python API. This code is not tested yet.

TODO: this function has not been used for a long time. It needs to be updated.
"""
from ..geolens import *

import os
import time
# import pyexr
import numpy as np

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_vccimaging')
from mitsuba.core import (Float, UInt32, Vector2f, Vector3f, Mask, Ray3f, RayDifferential3f,
Thread, sample_shifted, sample_rgb_spectrum, is_monochromatic, is_rgb, is_polarized, DEBUG)
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock


# # =======================================================================
# # Render by local PSF kernels (has been moved to render_psf.py)
# # =======================================================================

# def PSF_kernels(lens, foc_dis, img, depth, layers=10):
#     lens.refocus(depth=foc_dis)
    
#     return


# def local_psf_render_high_res(input, psf, patch_size=[320, 480], kernel_size=11):
#     B, C, H, W = input.shape
#     img_render = torch.zeros_like(input)
#     for pi in range(int(np.ceil(H/patch_size[0]))):    # int function here is not accurate
#         for pj in range(int(np.ceil(W/patch_size[1]))):
#             low_i = pi * patch_size[0]
#             up_i = min((pi+1)*patch_size[0], H)
#             low_j = pj * patch_size[1]
#             up_j =  min((pj+1)*patch_size[1], W)

#             img_patch = input[:, :, low_i:up_i, low_j:up_j]
#             psf_patch = psf[:, low_i:up_i, low_j:up_j, :, :]

#             img_render[:, :, low_i:up_i, low_j:up_j] = local_psf_render(img_patch, psf_patch, kernel_size=kernel_size)
    
#     return img_render


# def local_psf_render(input, psf, kernel_size=11):
#     """ Blurs image with dynamic Gaussian blur.

#     Args:
#         input (Tensor): The image to be blurred (N, C, H, W).
#         psf (Tensor): Per pixel local PSFs (1, H, W, ks, ks)
#         kernel_size (int): Size of the PSFs. Defaults to 11.

#     Returns:
#         output (Tensor): Rendered image (N, C, H, W)
#     """
    
#     if len(input.shape) < 4:
#         input = input.unsqueeze(0)

#     b,c,h,w = input.shape
#     pad = int((kernel_size-1)/2)

#     # 1. pad the input with replicated values
#     inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
#     # 2. Create a Tensor of varying Gaussian Kernel
#     kernels = psf.reshape(-1, kernel_size, kernel_size)
#     kernels_rgb = torch.stack(c*[kernels], 1)
#     # 3. Unfold input
#     inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))   
#     # 4. Multiply kernel with unfolded
#     x1 = inp_unf.view(b,c,-1,h*w)
#     x2 = kernels_rgb.view(b, h*w, c, -1).permute(0, 2, 3, 1)
#     y = (x1*x2).sum(2)
#     # 5. Fold and return
#     return torch.nn.functional.fold(y,(h,w),(1,1))



# =======================================================================
# Render by ray tracing and volume rendering
# =======================================================================

def render_3D(lens, foc_dis, img, depth, layers=10):
    """ Treat RGB image and Depth map as 3D scene and render image.

    Args:
        img (_type_): _description_
        depth (_type_): _description_
    """
    lens.refocus(depth=foc_dis)
    rays = lens.sample_sensor(vpp=32)
    depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 1.0, ]
    


    lens.refocus_inf()
    # return I


def path_integrater(lens, ray, depth_map, rgb):
    """
    Args:
        ray (Ray): [spp, H, W, 3]
        depth ([H, W, 1] ndarray): [description]
        rgb ([H, W, C] ndarray): [description]
        scale (): scene R = depth * tan_fov [description]
    """
    
    H, W, _ = depth_map.shape
    spp = ray.o.shape[0]
    dnear = -depth_map.min()
    dfar = -depth_map.max()
    depths = np.linspace(dnear, dfar, num=30)
    # ray.propagate_to(dmin)

    depth_map = torch.tensor(-depth_map.transpose((2,0,1))).to(lens.device)
    rgb = torch.tensor(rgb.transpose((2,0,1))).to(lens.device)
    outputs = torch.full((3, spp, H, W), 0.).to(lens.device)
    outputs_mask = torch.full((spp, H, W), 0.).to(lens.device)

    # path integration
    for depth in depths:
        h, w = -depth * np.tan(lens.hfov) * 2, -depth * np.tan(lens.hfov) * 2 *W/H
        # scale = 1/lens.calc_magnification3(depth)
        # h, w = lens.sensor_size[0]*scale, lens.sensor_size[1]*scale
        
        # projection
        ray.propagate_to(depth)
        p = ray.o[...,:2]
        ray.ra = ray.ra * (torch.abs(p[...,0])<w/2) * (torch.abs(p[...,1])<h/2)
        u = torch.clamp((w/2+p[..., 0])/w*W, min=0, max=W-0.01)
        v = torch.clamp((h/2+p[..., 1])/h*H, min=0.01, max=H) 
        idx_i = H - v.ceil().long()
        idx_j = u.floor().long()

        # integration
        intersection_mask = (depth <= depth_map[..., idx_i, idx_j]).squeeze(0)
        outputs_mask += ray.ra * intersection_mask
        outputs += ray.ra * intersection_mask * rgb[...,idx_i,idx_j]
        ray.ra = ray.ra * ~intersection_mask

        # tmp=(torch.sum(outputs, axis=1) + 1e-6)/(torch.sum(intersection_mask, axis=0) + 1e-3)
        # save_image(tmp.cpu().unsqueeze(0)/255., f'test_render{depth}.png')


    output = (torch.sum(outputs, axis=1) + 1e-6)/(torch.sum(intersection_mask, axis=0) + 1e-3)
    return output




# =======================================================================
# Render by Mitsuba2
# =======================================================================

class MTSScene(PrettyPrinter):
    def __init__(self, filename):
        Thread.thread().file_resolver().append(os.path.dirname(filename))
        self.scene = load_file(filename)
        self.R_sensor, self.t_sensor = self._find_transformation(filename)

    
    def prepare_lens(self, lensgroup, pixel_size, film_size, room_scale=1., R=None, t=None):
        self.lensgroup = lensgroup
        if (R is None) or (t is None):
            R = self.R_sensor
            t = self.t_sensor
        self.lensgroup.prepare_mts(pixel_size, film_size, R, t, room_scale)
    
    
    def render(self, nps=1, spp=1, idx=0):
        """
        Rendering:
        - nps: number of passes
        - pps: number samples per pixel
        - idx: sensor index
        """
        # render
        time_a = time.time()
        img = 0.
        for i in range(nps):
            print('At pass = {}/{} ...'.format(i, nps))
            img_org = self._render(spp, idx).numpy()
            img = img_org if i == 0 else img + img_org
        time_b = time.time()
        print('Rendering finished; took {} seconds'.format(time_b - time_a))
        # if save:
        #     pyexr.write("out.exr", img)
        #     np.save('img.npy', img)
        return img.reshape(self.lensgroup.film_size[1], self.lensgroup.film_size[0], -1)
    
    
    def _sample_ray(self, sample1, sample2, sample3, time=0):
        # sample wavelength by Mitsuba2
        wav, weights = sample_rgb_spectrum(sample_shifted(sample1))

        # ray-traced {o,d} from the lensgroup
        o, d, valid = self.lensgroup.sample_ray_mts(int(len(wav.x)), wav, sample2, sample3)

        # return Mitsuba2 rays
        active = Mask(valid.numpy())
        rays = RayDifferential3f(Ray3f(o, d, 1e-5, 1e5, time, wav))
        return rays, weights, active

    
    def trace(self, spp=None, sensor_index=0):
        """
        Internally used function: render the specified Mitsuba scene and return a
        floating point array containing RGB values and AOVs, if applicable
        """
        sensor = self.scene.sensors()[sensor_index]
        sampler = sensor.sampler()
        film_size = self.lensgroup.film_size
        if spp is None:
            spp = sampler.sample_count()

        total_sample_count = ek.hprod(film_size) * spp

        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = ek.arange(UInt32, total_sample_count)
        pos //= spp
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(Float(pos % int(film_size[0])), Float(pos // int(film_size[0])))
        pos += sampler.next_2d()

        # samples
        sample1 = sampler.next_1d()
        sample2 = pos * scale
        sample3 = sampler.next_2d()

        wav, weights = sample_rgb_spectrum(sample_shifted(sample1))

        # ray-traced {o,d} from the lensgroup
        o, d, valid = self.lensgroup.sample_ray_mts(int(len(wav.x)), wav, sample2, sample3)

        o, d, valid = (np.squeeze(v.reshape(
            self.lensgroup.film_size[1], self.lensgroup.film_size[0], -1))
        for v in [o,d,valid])
        return o, d, valid.numpy()
        
    
    def _render(self, spp=None, sensor_index=0):
        """
        Internally used function: render the specified Mitsuba scene and return a
        floating point array containing RGB values and AOVs, if applicable
        """
        sensor = self.scene.sensors()[sensor_index]
        sampler = sensor.sampler()
        film_size = self.lensgroup.film_size
        if spp is None:
            spp = sampler.sample_count()

        total_sample_count = ek.hprod(film_size) * spp

        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = ek.arange(UInt32, total_sample_count)
        pos //= spp
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(Float(pos % int(film_size[0])), Float(pos // int(film_size[0])))
        pos += sampler.next_2d()

        rays, weights, active = self._sample_ray(
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=sampler.next_2d(),
            time=0
        )

        spec, mask, aovs = self.scene.integrator().sample(self.scene, sampler, rays, active=active)
        spec *= weights
        del mask

        if is_polarized:
            from mitsuba.core import depolarize
            spec = depolarize(spec)

        if is_monochromatic:
            rgb = [spec[0]]
        elif is_rgb:
            rgb = spec
        else:
            from mitsuba.core import spectrum_to_xyz, xyz_to_srgb
            xyz = spectrum_to_xyz(spec, rays.wavelengths)
            rgb = xyz_to_srgb(xyz)
            del xyz

        aovs.insert(0, Float(1.0))
        for i in range(len(rgb)):
            aovs.insert(i + 1, rgb[i])
        del rgb, spec, weights, rays
        
        block = ImageBlock(
            size=film_size,
            channel_count=len(aovs),
            filter=sensor.film().reconstruction_filter(),
            warn_negative=False,
            warn_invalid=DEBUG,
            border=False
        )

        block.clear()
        block.put(pos, aovs)

        del pos
        del aovs

        data = block.data()

        ch = block.channel_count()
        i = UInt32.arange(ek.hprod(block.size()) * (ch - 1))

        weight_idx = i // (ch - 1) * ch
        values_idx = (i * ch) // (ch - 1) + 1

        weight = ek.gather(data, weight_idx)
        values = ek.gather(data, values_idx)

        return values / (weight + 1e-8)


    @staticmethod
    def _find_transformation(filename):
        with open(filename, 'r') as file:
            data = file.read()
        a = data.find('<sensor')
        b = data.find('</sensor')
        data = data[a:b]
        c = data.find('<matrix value=')
        d = data.find('</transform>')
        T = []
        for v in data[c+14:d].split():
            num_str = ''.join((c if c in '0123456789e-.' else '' for c in v))
            T.append(float(num_str))
        T = np.array(T).reshape((4,4))
        R, t = T[:3,:3], T[:3,3]
        return R, t

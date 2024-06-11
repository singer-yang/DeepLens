""" Forward and backward Monte-Carlo integral functions.
"""
import torch
import numpy as np
import torch.nn.functional as nnF

from .basics import EPSILON

def backward_integral(ray, img, ps, H, W, interpolate=True, pad=True, energy_correction=1):
    """ Backward integral, for ray tracing based rendering.

    Ignore:
        1. sub-pixel phase shiftment
        2. ray obliquity energy decay

        If we want to use this correction terms, use energy_corrention variable.

    Args:
        ray: Ray object. Shape of ray.o is [spp, 1, 3].
        img: [B, C, H, W]
        ps: pixel size
        H: image height
        W: image width
        interpolate: whether to interpolate
        pad: whether to pad the image
        energy_correction: whether to keep incident and output image total energy unchanged

    Returns:
        output: shape [B, C, H, W]
    """
    assert len(img.shape) == 4
    p = ray.o[...,:2]

    if pad:
        img = nnF.pad(img, (1,1,1,1), "replicate")

        # Convert ray positions to uv coordinates
        u = torch.clamp(W/2 + p[..., 0] / ps, min=-0.99, max=W-0.01)
        v = torch.clamp(H/2 + p[..., 1] / ps, min=0.01, max=H+0.99) 

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1
    else:
        # Convert ray positions to uv coordinates
        u = torch.clamp(W/2 + p[..., 0] / ps, min=0.01, max=W-1.01)
        v = torch.clamp(H/2 + p[..., 1] / ps, min=1.01, max=H-0.01)

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long()
        idx_j = u.floor().long()

    # gradients are stored in weight parameters
    w_i = v - v.floor().long()
    w_j = u.ceil().long() - u

    if ray.coherent:
        raise Exception('Backward coherent integral needs to be checked.')
        
    else:
        if interpolate: # Bilinear interpolation
            # img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], out_img shape [N, C, spp, H, W]
            out_img =  img[...,idx_i, idx_j] * w_i * w_j
            out_img += img[...,idx_i+1, idx_j] * (1-w_i) * w_j
            out_img += img[...,idx_i, idx_j+1] * w_i * (1-w_j)
            out_img += img[...,idx_i+1, idx_j+1] * (1-w_i) * (1-w_j)

        else:
            out_img =  img[...,idx_i, idx_j]

        # Monte-Carlo integration
        output = (torch.sum(out_img * ray.ra * energy_correction, -3) + 1e-9) / (torch.sum(ray.ra, -3) + 1e-6)
        return output


def forward_integral(ray, ps, ks, pointc_ref=None, coherent=False):
    """ Forward integral model, including PSF and vignetting

    Common bugs:
        1, When you got index out of range, it is usually a memory issue.
        2, 

    Args:
        ray: Ray object. Shape of ray.o is [spp, N, 3].
        ps: pixel size
        ks: kernel size.
        pointc_ref: reference pointc, shape [2]
        center: whether to center the PSF.
        interpolate: whether to interpolate the PSF

    Returns:
        psf: point spread function, shape [N, ks, ks]
    """
    single_point = True if len(ray.o.shape) == 2 else False
    points = - ray.o[..., :2]       # shape [spp, N, 2] or [spp, 2]. flip points.
    psf_range = [(- ks / 2 + 0.5) * ps, (ks / 2 - 0.5) * ps]    # this ensures the pixel size doesnot change in assign_points_to_pixels function
    
    # ==> PSF center
    if pointc_ref is None:
        # Use RMS center
        pointc = (points * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON)
        points_shift = points - pointc
    else:
        # Use manually given center (can be calculated by chief ray or perspective)
        points_shift = points - pointc_ref.to(points.device)
    
    # ==> Remove invalid points
    ra = ray.ra * (points_shift[...,0].abs() < (psf_range[1] - 0.1 * ps)) * (points_shift[...,1].abs() < (psf_range[1] - 0.1 * ps))   # shape [spp, N] or [spp].
    points_shift = points_shift * ra.unsqueeze(-1)
    
    # ==> Monte Carlo integral
    # Incoherent ray tracing, integral over intensity
    if not coherent: 
        if single_point:
            obliq = ray.d[:, 2]**2
            psf = assign_points_to_pixels(points=points_shift, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra, obliq=obliq)
        else:
            psf = []
            for i in range(ray.o.shape[1]):
                points_shift0 = points_shift[:, i, :]   # from [spp, N, 2] to [spp, 2]
                ra0 = ra[:, i]                          # from [spp, N] to [spp]
                obliq = ray.d[:, i, 2]**2
                
                psf0 = assign_points_to_pixels(points=points_shift0, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra0, obliq=obliq)
                psf.append(psf0)
            
            psf = torch.stack(psf, dim=0)   # shape [N, ks, ks]
    
    # Coherent ray tracing, integral over complex amplitude
    else:
        if single_point:
            obliq = ray.d[:, 2]     # from [spp, 3] to [spp], for amplitude correction. 
            opl = ray.opl[:, 0]     # from [spp, 1] to [spp]
            phase = torch.fmod((opl - opl.min(0).values) / (ray.wvln * 1e-3), 1) * (2 * np.pi) # [spp, N] to [spp]
            
            psf = assign_points_to_pixels(points=points_shift, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra, coherent=True, phase=phase, d=ray.d[:, i, :], obliq=obliq)
        else:
            psf = []
            for i in range(ray.o.shape[1]):
                points_shift0 = points_shift[:, i, :]   # from [spp, N, 2] to [spp, 2]
                ra0 = ra[:, i]                          # from [spp, N] to [spp]
                obliq = ray.d[:, i, 2]      # from [spp, N, 3] to [spp], for amplitude correction. 
                opl = ray.opl[:, i]         # from [spp, N] to [spp]
                phase = torch.fmod((opl - opl.min(0).values) / (ray.wvln * 1e-3), 1) * (2 * np.pi) # [spp, N] to [spp]
                
                psf_u = assign_points_to_pixels(points=points_shift0, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra0, coherent=True, phase=phase, d=ray.d[:, i, :], obliq=obliq)
                psf.append(psf_u)
        
            psf = torch.stack(psf, dim=0)   # shape [N, ks, ks]
    
    return psf


def assign_points_to_pixels(points, ks, x_range, y_range, ra, interpolate=True, coherent=False, phase=None, d=None, obliq=None, sub_pixel_phase=False, wvln=0.589):
    """ Assign points to pixels, both coherent and incoherent. Use advanced indexing to increment the count for each corresponding pixel. This function can only compute single point source, single wvln. If you want to compute multiple point or muyltiple wvln, please call this function multiple times.
    
    Args:
        points: shape [spp, 1, 2]
        ks: kernel size
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
        ra: shape [spp, 1, 1]
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp, 1, 1]

    Returns:
        psf: shape [ks, ks]
    """
    # ==> Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range
    ps = (x_max - x_min) / (ks - 1)

    # ==> Normalize points to the range [0, 1]
    points_normalized = torch.zeros_like(points)
    points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
    points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

    if interpolate:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
        pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
        pixel_indices_br = pixel_indices_tl + 1

        
        # ==> Use advanced indexing to increment the count for each corresponding pixel
        if coherent:
            grid = torch.zeros(ks, ks).to(device) + 0j
            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*obliq*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*obliq*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*obliq*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*obliq*torch.exp(1j*phase), accumulate=True)

        else:
            grid = torch.zeros(ks, ks).to(points.device)
            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*obliq, accumulate=True)

    else:
        pixel_indices_float = points_normalized * (ks - 1)
        pixel_indices_tl = pixel_indices_float.floor().long()

        if coherent:
            raise Warning("Need to check sub-pixel phase shift.")

        else:
            grid = torch.zeros(ks, ks).to(points.device)
            grid.index_put_(tuple(pixel_indices_tl.t()), ra, accumulate=True)
        
    return grid



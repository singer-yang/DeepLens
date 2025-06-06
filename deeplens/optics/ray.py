"""Optical ray class.

Copyright (c) 2025 Xinge Yang (xinge.yang@kaust.edu.sa)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import copy

import torch
import torch.nn.functional as F

from .basics import DEFAULT_WAVE, EPSILON, DeepObj


class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device="cpu"):
        """Optical ray class. 
        
        Now we only support the same wvln for all rays, but it is possible to extend to different wvln for each ray.
        """
        assert wvln > 0.1 and wvln < 1, "wvln should be in [um]"
        self.wvln = wvln

        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.ra = torch.ones(o.shape[:-1])
        self.valid = self.ra.clone()

        # Intensity tracing
        self.en = torch.ones(o.shape[:-1])

        # Coherent ray tracing (initialize coherent light)
        self.coherent = coherent  # bool
        self.opl = torch.zeros(o.shape[:-1])

        # Used in lens design with no direct physical meaning
        self.obliq = torch.ones(o.shape[:-1])

        self.to(device)

        # Post computation
        self.d = F.normalize(self.d, p=2, dim=-1)
        self.is_forward = bool((self.d[..., 2] > 0).any())

    def prop_to(self, z, n=1):
        """Ray propagates to a given depth plane.

        Args:
            z (float): depth.
            n (float, optional): refractive index. Defaults to 1.
        """
        t = (z - self.o[..., 2]) / self.d[..., 2]
        new_o = self.o + self.d * t[..., None]

        is_valid = (self.ra > 0) & (torch.abs(t) >= 0)
        new_o[~is_valid] = self.o[~is_valid]
        self.o = new_o

        if self.coherent:
            if t.min() > 100 and torch.get_default_dtype() == torch.float32:
                raise Warning(
                    "Should use float64 in coherent ray tracing for precision."
                )
            else:
                self.opl = self.opl + n * t

        return self

    def centroid(self):
        """Calculate the centroid of the ray, shape (..., num_rays, 3)

        Returns:
            torch.Tensor: Centroid of the ray, shape (..., 3)
        """
        return (self.o * self.ra.unsqueeze(-1)).sum(-2) / self.ra.sum(-1).add(
            EPSILON
        ).unsqueeze(-1)
    
    def rms_error(self, center_ref=None):
        """Calculate the RMS error of the ray.

        Args:
            center_ref (torch.Tensor): Reference center of the ray, shape (..., 3). If None, use the centroid of the ray as reference.
        Returns:
            torch.Tensor: average RMS error of the ray
        """
        # Calculate the centroid of the ray as reference
        if center_ref is None:
            with torch.no_grad():
                center_ref = self.centroid()

        center_ref = center_ref.unsqueeze(-2)
        
        # Calculate RMS error for each region
        rms_error = ((self.o[..., :2] - center_ref[..., :2])**2).sum(-1)
        rms_error = (rms_error * self.ra).sum(-1) / (self.ra.sum(-1) + EPSILON)
        rms_error = rms_error.sqrt()
        
        # Average RMS error
        return rms_error.mean()

    def clone(self, device=None):
        """Clone the ray.

        Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        if device is None:
            return copy.deepcopy(self).to(self.device)
        else:
            return copy.deepcopy(self).to(device)

    def squeeze(self, dim=None):
        """Squeeze the ray.

        Args:
            dim (int, optional): dimension to squeeze. Defaults to None.
        """
        self.o = self.o.squeeze(dim)
        self.d = self.d.squeeze(dim)
        self.ra = self.ra.squeeze(dim)
        self.valid = self.valid.squeeze(dim)
        self.en = self.en.squeeze(dim)
        self.opl = self.opl.squeeze(dim)
        self.obliq = self.obliq.squeeze(dim)
        return self

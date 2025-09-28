# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Optical ray class."""

import copy

import torch
import torch.nn.functional as F

from deeplens.optics.basics import DEFAULT_WAVE, EPSILON, DeepObj


class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device="cpu"):
        """Optical ray class."""

        # Basic ray parameters
        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.shape = self.o.shape[:-1]
        assert wvln > 0.1 and wvln < 1, "Ray wavelength unit should be [um]"
        self.wvln = torch.full((*self.shape, 1), wvln)

        # Auxiliary ray parameters
        self.valid = torch.ones(self.shape)
        self.en = torch.ones((*self.shape, 1))
        self.obliq = torch.ones((*self.shape, 1))
        self.is_forward = self.d[..., 2].unsqueeze(-1) > 0

        # Coherent ray tracing
        self.coherent = coherent  # bool
        self.opl = torch.zeros((*self.shape, 1))

        self.to(device)
        self.d = F.normalize(self.d, p=2, dim=-1)

    def prop_to(self, z, n=1.0):
        """Ray propagates to a given depth plane.

        Args:
            z (float): depth.
            n (float, optional): refractive index. Defaults to 1.
        """
        t = (z - self.o[..., 2]) / self.d[..., 2]
        new_o = self.o + self.d * t.unsqueeze(-1)
        valid_mask = (self.valid > 0).unsqueeze(-1)
        self.o = torch.where(valid_mask, new_o, self.o)

        if self.coherent:
            if t.abs().max() > 100.0 and t.dtype != torch.float64:
                raise Warning("Should use float64 in coherent ray tracing.")
            else:
                new_opl = self.opl + n * t.unsqueeze(-1)
                self.opl = torch.where(valid_mask, new_opl, self.opl)

        return self

    def centroid(self):
        """Calculate the centroid of the ray, shape (..., num_rays, 3)

        Returns:
            torch.Tensor: Centroid of the ray, shape (..., 3)
        """
        return (self.o * self.valid.unsqueeze(-1)).sum(-2) / self.valid.sum(-1).add(
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
        rms_error = (rms_error * self.valid).sum(-1) / (self.valid.sum(-1) + EPSILON)
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
        self.wvln = self.wvln.squeeze(dim)
        self.valid = self.valid.squeeze(dim)
        self.en = self.en.squeeze(dim)
        self.opl = self.opl.squeeze(dim)
        self.obliq = self.obliq.squeeze(dim)
        self.is_forward = self.is_forward.squeeze(dim)
        return self

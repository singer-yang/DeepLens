# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Optical ray class."""

import copy

import torch
import torch.nn.functional as F

from deeplens.basics import DEFAULT_WAVE, EPSILON, DeepObj


class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device="cpu"):
        """Initialize a ray object.

        Args:
            o (torch.Tensor): Ray origin, shape (*batch_size, num_rays, 3).
            d (torch.Tensor): Ray direction, shape (*batch_size, num_rays, 3).
            wvln (float or torch.Tensor): Ray wavelength, unit: [um]. If a tensor, shape must be (*batch_size, 1).
            coherent (bool): Whether to use coherent ray tracing.
            device (str): Device to store the ray.
        """
        # Basic ray parameters
        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.shape = self.o.shape[:-1]

        # Wavelength
        assert wvln > 0.1 and wvln < 10.0, "Ray wavelength unit should be [um]"
        self.wvln = torch.tensor(wvln)

        # Auxiliary ray parameters
        self.is_valid = torch.ones(self.shape)
        self.en = torch.ones((*self.shape, 1))
        self.obliq = torch.ones((*self.shape, 1))

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
        valid_mask = (self.is_valid > 0).unsqueeze(-1)
        self.o = torch.where(valid_mask, new_o, self.o)

        if self.coherent:
            if t.dtype != torch.float64:
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
        return (self.o * self.is_valid.unsqueeze(-1)).sum(-2) / self.is_valid.sum(
            -1
        ).add(EPSILON).unsqueeze(-1)

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
        rms_error = ((self.o[..., :2] - center_ref[..., :2]) ** 2).sum(-1)
        rms_error = (rms_error * self.is_valid).sum(-1) / (
            self.is_valid.sum(-1) + EPSILON
        )
        rms_error = rms_error.sqrt()

        # Average RMS error
        return rms_error.mean()

    def flip_xy(self):
        """Flip the x and y coordinates of the ray.

        This function is used when calculating point spread function and wavefront distribution.
        """
        self.o = torch.cat([-self.o[..., :2], self.o[..., 2:]], dim=-1)
        self.d = torch.cat([-self.d[..., :2], self.d[..., 2:]], dim=-1)
        return self

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
        # wvln is a single element tensor, no squeeze needed
        self.is_valid = self.is_valid.squeeze(dim)
        self.en = self.en.squeeze(dim)
        self.opl = self.opl.squeeze(dim)
        self.obliq = self.obliq.squeeze(dim)
        return self

    def unsqueeze(self, dim=None):
        """Unsqueeze the ray.

        Args:
            dim (int, optional): dimension to unsqueeze. Defaults to None.
        """
        self.o = self.o.unsqueeze(dim)
        self.d = self.d.unsqueeze(dim)
        # wvln is a single element tensor, no unsqueeze needed
        self.is_valid = self.is_valid.unsqueeze(dim)
        self.en = self.en.unsqueeze(dim)
        self.opl = self.opl.unsqueeze(dim)
        self.obliq = self.obliq.unsqueeze(dim)
        return self

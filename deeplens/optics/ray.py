"""Optical ray"""

import copy

import torch
import torch.nn.functional as F

from .basics import DEFAULT_WAVE, EPSILON, DeepObj


class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device="cpu"):
        """Ray class. Optical rays with the same wvln.

        Args:
            o (torch.Tensor): origin of the ray.
            d (torch.Tensor): direction of the ray.
            wvln (float, optional): wavelength in [um]. Defaults to DEFAULT_WAVE.
            coherent (bool, optional): coherent ray tracing. Defaults to False.
            device (str, optional): device. Defaults to "cpu".
        """
        assert wvln > 0.1 and wvln < 1, "wvln should be in [um]"
        self.wvln = wvln

        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.ra = torch.ones(o.shape[:-1])
        self.valid = self.ra

        # Intensity tracing
        self.en = torch.ones(o.shape[:-1])

        # Coherent ray tracing (initialize coherent light)
        self.coherent = coherent
        self.opl = torch.zeros(o.shape[:-1])

        # Used in lens design with no direct physical meaning
        self.obliq = torch.ones(o.shape[:-1])

        self.to(device)
        self.d = F.normalize(self.d, p=2, dim=-1)

    def prop_to(self, z, n=1):
        """Ray propagates to a given depth plane.

        Args:
            z (float): depth.
            n (float, optional): refractive index. Defaults to 1.
        """
        return self.propagate_to(z, n)

    def propagate_to(self, z, n=1):
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

    def project_to(self, z):
        """Calculate the intersection point of the ray with a given depth plane.

        Args:
            z (float): depth.

        Return:
            p: shape of [..., 2].
        """
        t = (z - self.o[..., 2]) / self.d[..., 2]
        new_o = self.o + self.d * t[..., None]
        is_valid = (self.ra > 0) & (torch.abs(t) >= 0)
        new_o[~is_valid] = self.o[~is_valid]
        return new_o[..., :2]

    def clone(self, device=None):
        """Clone the ray.

        Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        if device is None:
            return copy.deepcopy(self).to(self.device)
        else:
            return copy.deepcopy(self).to(device)

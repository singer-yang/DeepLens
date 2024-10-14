"""Optical ray class."""

import copy

import torch
import torch.nn.functional as F

from .basics import DEFAULT_WAVE, DEVICE, DeepObj


class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device=DEVICE):
        """Ray class. Optical rays with the same wvln.

        Args:
            o (Tensor): ray position. shape [..., 3]
            d (Tensor): normalized ray direction. shape [..., 3]
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            ra (Tensor, optional): Validity. Defaults to None.
            en (Tensor, optional): Spherical wave energy decay. Defaults to None.
            obliq (Tensor, optional): Obliquity energy decay, now only used to record refractive angle. Defaults to None.
            opl (Tensor, optional): Optical path length, Now used as the phase term. Defaults to None.
            coherent (bool, optional): If the ray is coherent. Defaults to False.
            device (torch.device, optional): Defaults to torch.device('cuda:0').
        """
        assert wvln > 0.1 and wvln < 1, "wvln should be in [um]"
        self.wvln = wvln

        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.ra = torch.ones(o.shape[:-1])

        # not used
        self.en = torch.ones(o.shape[:-1])

        # used in coherent ray tracing
        self.coherent = coherent
        self.opl = torch.zeros(o.shape[:-1])

        # used in lens design
        self.obliq = torch.ones(o.shape[:-1])

        self.to(device)
        self.d = F.normalize(self.d, p=2, dim=-1)

    def prop_to(self, z, n=1):
        """Ray propagates to a given depth."""
        return self.propagate_to(z, n)

    def propagate_to(self, z, n=1):
        """Ray propagates to a given depth.

        Args:
            z (float): depth.
            n (float, optional): refractive index. Defaults to 1.
        """
        o0 = self.o.clone()
        t = (z - self.o[..., 2]) / self.d[..., 2]
        self.o = self.o + self.d * t[..., None]

        if self.coherent:
            if t.min() > 100 and torch.get_default_dtype() == torch.float32:
                raise Warning("Should use float64 in coherent ray tracing.")
            else:
                self.opl = self.opl + n * t

        return self

    def project_to(self, z):
        """Calculate the intersection points of ray with plane z.

        Return:
            p: shape of [..., 2].
        """
        t = (z - self.o[..., 2]) / self.d[..., 2]
        p = self.o[..., 0:2] + self.d[..., 0:2] * t[..., None]
        return p

    def clone(self, device=None):
        """Clone a Ray.

        Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        if device is None:
            return copy.deepcopy(self).to(self.device)
        else:
            return copy.deepcopy(self).to(device)

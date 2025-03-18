"""Optical ray"""

import copy

import torch
import torch.nn.functional as F

from .basics import DEFAULT_WAVE, DeepObj


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

    def init_polarization(self, mode="incoherent"):
        """Initialize the polarization of the ray.

        Polarization is a property of transverse waves. In linear polarization, the (electric) fields oscillate in a single direction. In circular or elliptical polarization, the (electric) fields rotate in a plane as the wave travels.

        Difference between coherent ray tracing and polarization ray tracing: coherent ray tracing is a simpler model compared to polarization ray tracing.

        We decompose the electric field into two orthogonal components, Es and Ep. When dealing with Fresnel equation, we first project Es and Ep to the plane of incidence, and then calculate the reflection and transmission (new Es and Ep).

        Reference:
            https://en.wikipedia.org/wiki/Polarization_(waves)
        """
        if mode == "incoherent":
            # Incoherent states can be modeled stochastically as a weighted combination of such uncorrelated waves with some distribution of frequencies (its spectrum), phases, and polarizations.

            # Random directions for the cross product
            d_random = self.d + torch.randn_like(self.d) * 0.1

            # Only initialize Es
            Es_d = torch.cross(self.d, d_random, dim=-1)
            self.Es_d = F.normalize(Es_d, p=2, dim=-1)
            self.Es = torch.ones_like(self.o[..., 0]) + 0j  # Es is complex scalar

            # Ep only becomes meanlingful after one Fresnel reflection/refraction
            Ep_d = torch.cross(self.d, self.Es_d, dim=-1)
            self.Ep_d = F.normalize(Ep_d, p=2, dim=-1)
            self.Ep = torch.zeros_like(self.o[..., 0]) + 0j

        elif mode == "linear":
            # A random direction for the cross product
            d_random = (
                self.d + torch.randn(3).unsqueeze(0).broadcast_to(self.d.shape) * 0.1
            )

            # Only initialize Es
            Es_d = torch.cross(self.d, d_random, dim=-1)
            self.Es_d = F.normalize(Es_d, p=2, dim=-1)
            self.Es = torch.ones_like(self.o[..., 0]) + 0j  # Es is complex scalar

            # Ep only becomes meanlingful after one Fresnel reflection/refraction
            Ep_d = torch.cross(self.d, self.Es_d, dim=-1)
            self.Ep_d = F.normalize(Ep_d, p=2, dim=-1)
            self.Ep = torch.zeros_like(self.o[..., 0]) + 0j

        elif mode == "circular":
            # A random direction for the cross product
            d_random = (
                self.d + torch.randn(3).unsqueeze(0).broadcast_to(self.d.shape) * 0.1
            )

            # Initialize Es and Ep
            Es_d = torch.cross(self.d, d_random, dim=-1)
            self.Es_d = F.normalize(Es_d, p=2, dim=-1)
            self.Es = (
                torch.ones_like(self.o[..., 0].unsqueeze(-1)) + 0j
            )  # shape of [N, 1]

            Ep_d = torch.cross(self.d, self.Es_d, dim=-1)
            self.Ep_d = F.normalize(Ep_d, p=2, dim=-1)

            # Phase difference between Es and Ep
            self.phi_sp = torch.ones_like(self.o[..., 0].unsqueeze(-1)) * torch.pi / 2
            self.Ep = (torch.ones_like(self.o[..., 0].unsqueeze(-1)) + 0j) * torch.exp(
                1j * self.phi_sp
            )

            self.Es = self.Es / (self.Es + self.Ep).abs()
            self.Ep = self.Ep / (self.Es + self.Ep).abs()

        else:
            raise NotImplementedError

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
        self.o = torch.where(self.ra[..., None] == 1, new_o, self.o)

        if self.coherent:
            if t.min() > 100 and torch.get_default_dtype() == torch.float32:
                raise Warning("Should use float64 in coherent ray tracing for precision.")
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
        p = self.o[..., :2] + self.d[..., :2] * t[..., None]
        return p

    def clone(self, device=None):
        """Clone the ray.

        Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        if device is None:
            return copy.deepcopy(self).to(self.device)
        else:
            return copy.deepcopy(self).to(device)

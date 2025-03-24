"""An ideal thin lens without any chromatic aberration."""

import torch
import torch.nn.functional as F
from .base import DiffractiveSurface


class ThinLens(DiffractiveSurface):
    def __init__(
        self,
        d,
        size,
        f0=None,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        device="cpu",
    ):
        """Initialize a thin lens.

        Args:
            d (float): Distance of the DOE surface. [mm]
            size (tuple or int): Size of the DOE, [w, h]. [mm]
            f0 (float): Initial focal length. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            mat (str): Material of the DOE.
            fab_ps (float): Fabrication pixel size. [mm]
            device (str): Device to run the DOE.
        """
        super().__init__(d=d, size=size, res=res, mat=mat, fab_ps=fab_ps, device=device)

        # Initial focal length
        if f0 is None:
            self.f0 = (
                torch.randn(1, device=self.device) * 1e6
            )  # [mm], initial a very large focal length
        else:
            self.f0 = torch.tensor(f0, device=self.device)

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a thin lens from a dict."""
        d = doe_dict["d"]
        size = doe_dict["size"]
        f0 = doe_dict.get("f0", None)
        res = doe_dict.get("res", (2000, 2000))
        return cls(
            d=d,
            size=size,
            f0=f0,
            res=res,
        )

    def get_phase_map(self, wvln=0.55):
        """Get the phase map at the given wavelength."""

        # Same focal length for all wavelengths
        wvln_mm = wvln * 1e-3
        phase_map = -2*torch.pi * (self.x**2 + self.y**2) / (2 * self.f0 * wvln_mm)
        phase_map = torch.remainder(phase_map, 2 * torch.pi)

        # Interpolate to the desired resolution
        phase_map = (
            F.interpolate(
                phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        return phase_map

    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self):
        """Activate gradients for optimization."""
        self.f0.requires_grad = True

    def get_optimizer_params(self, lr=0.1):
        """Get parameters for optimization."""
        self.activate_grad()
        return [{"params": [self.f0], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["f0"] = self.f0.item()
        return surf_dict

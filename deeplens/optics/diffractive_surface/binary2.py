"""Binary2 DOE parameterization."""

import torch
from .base import DiffractiveSurface


class Binary2(DiffractiveSurface):
    def __init__(self, d, size, res=(2000, 2000), mat="fused_silica", fab_ps=0.001, device="cpu"):
        """Initialize Binary DOE."""
        super().__init__(d=d, size=size, res=res, mat=mat, fab_ps=fab_ps, device=device)

        # Initialize with random small values
        self.alpha2 = (torch.rand(1) - 0.5) * 0.02
        self.alpha4 = (torch.rand(1) - 0.5) * 0.002
        self.alpha6 = (torch.rand(1) - 0.5) * 0.0002
        self.alpha8 = (torch.rand(1) - 0.5) * 0.00002
        self.alpha10 = (torch.rand(1) - 0.5) * 0.000002

        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w/2, self.w/2, self.res[1]),
            torch.linspace(self.h/2, -self.h/2, self.res[0]),
            indexing="xy",
        )

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Binary DOE from a dict."""
        d = doe_dict["d"]
        size = doe_dict["size"]
        res = doe_dict.get("res", (2000, 2000))
        fab_ps = doe_dict.get("fab_ps", 0.001)
        return cls(
            size=size,
            d=d,
            res=res,
            fab_ps=fab_ps,
        )

    def _phase_map0(self):
        """Get the phase map at design wavelength."""
        # Calculate radial distance
        r2 = self.x**2 + self.y**2

        # Calculate phase using Binary DOE formula
        phase = torch.pi * (
            self.alpha2 * r2
            + self.alpha4 * r2**2
            + self.alpha6 * r2**3
            + self.alpha8 * r2**4
            + self.alpha10 * r2**5
        )

        return phase
    


    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self):
        """Activate gradients for optimization."""
        self.alpha2.requires_grad = True
        self.alpha4.requires_grad = True
        self.alpha6.requires_grad = True
        self.alpha8.requires_grad = True
        self.alpha10.requires_grad = True

    def get_optimizer_params(self, lr=0.001):
        """Get parameters for optimization."""
        self.activate_grad()
        return [
            {
                "params": [
                    self.alpha2,
                    self.alpha4,
                    self.alpha6,
                    self.alpha8,
                    self.alpha10,
                ],
                "lr": lr,
            }
        ]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["alpha2"] = round(self.alpha2.item(), 6)
        surf_dict["alpha4"] = round(self.alpha4.item(), 6)
        surf_dict["alpha6"] = round(self.alpha6.item(), 6)
        surf_dict["alpha8"] = round(self.alpha8.item(), 6)
        surf_dict["alpha10"] = round(self.alpha10.item(), 6)
        return surf_dict

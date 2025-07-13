"""Binary2 DOE parameterization."""

import torch
from deeplens.optics.diffractive_surface.diffractive import DiffractiveSurface


class Binary2(DiffractiveSurface):
    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        device="cpu",
    ):
        """Initialize Binary DOE."""
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, device=device
        )

        # Initialize with random small values
        self.alpha2 = (torch.rand(1) - 0.5) * 0.02
        self.alpha4 = (torch.rand(1) - 0.5) * 0.002
        self.alpha6 = (torch.rand(1) - 0.5) * 0.0002
        self.alpha8 = (torch.rand(1) - 0.5) * 0.00002
        self.alpha10 = (torch.rand(1) - 0.5) * 0.000002

        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
            indexing="xy",
        )

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Binary DOE from a dict."""
        d = doe_dict["d"]
        res = doe_dict.get("res", (2000, 2000))
        fab_ps = doe_dict.get("fab_ps", 0.001)
        wvln0 = doe_dict.get("wvln0", 0.55)
        mat = doe_dict.get("mat", "fused_silica")
        return cls(
            d=d,
            res=res,
            mat=mat,
            wvln0=wvln0,
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
    def get_optimizer_params(self, lr=0.001):
        """Get parameters for optimization.

        Args:
            lr (float): Base learning rate for alpha2. Learning rates for higher-order parameters will be scaled progressively (10x, 100x, 1000x, 10000x).
        """
        self.alpha2.requires_grad = True
        self.alpha4.requires_grad = True
        self.alpha6.requires_grad = True
        self.alpha8.requires_grad = True
        self.alpha10.requires_grad = True

        optimizer_params = [
            {"params": [self.alpha2], "lr": lr},
            {"params": [self.alpha4], "lr": lr * 10},
            {"params": [self.alpha6], "lr": lr * 100},
            {"params": [self.alpha8], "lr": lr * 1000},
            {"params": [self.alpha10], "lr": lr * 10000},
        ]

        return optimizer_params

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

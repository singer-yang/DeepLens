"""Pixel2D DOE parameterization."""

import torch
from .base import DiffractiveSurface


class Pixel2D(DiffractiveSurface):
    """Pixel2D DOE parameterization - direct phase map representation."""

    def __init__(
        self,
        d,
        size,
        phase_map_path=None,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        device="cpu",
    ):
        """Initialize Pixel2D DOE, where each pixel is independent parameter.

        Args:
            d (float): Distance of the DOE surface. [mm]
            size (tuple or int): Size of the DOE, [w, h]. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            mat (str): Material of the DOE.
            fab_ps (float): Fabrication pixel size. [mm]
            device (str): Device to run the DOE.
        """
        super().__init__(d=d, size=size, res=res, mat=mat, fab_ps=fab_ps, device=device)

        # Initialize phase map with random values
        if phase_map_path is None:
            self.phase_map = torch.randn(self.res, device=self.device) * 1e-3

        elif isinstance(phase_map_path, str):
            self.phase_map = torch.load(phase_map_path, map_location=device)

        else:
            raise ValueError(f"Invalid phase_map_path: {phase_map_path}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Pixel2D DOE from a dict."""
        size = doe_dict["size"]
        d = doe_dict["d"]
        res = doe_dict.get("res", (2000, 2000))
        fab_ps = doe_dict.get("fab_ps", 0.001)
        phase_map_path = doe_dict.get("phase_map_path", None)
        return cls(
            size=size,
            d=d,
            res=res,
            fab_ps=fab_ps,
            phase_map_path=phase_map_path,
        )

    def _phase_map0(self):
        """Get the phase map at design wavelength."""
        return self.phase_map

    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self):
        """Activate gradients for optimization."""
        self.phase_map.requires_grad = True

    def get_optimizer_params(self, lr=None):
        """Get parameters for optimization."""
        self.activate_grad()
        lr = 0.01 if lr is None else lr
        return [{"params": [self.phase_map], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, phase_map_path):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["phase_map_path"] = phase_map_path
        torch.save(self.phase_map.clone().detach().cpu(), phase_map_path)
        return surf_dict

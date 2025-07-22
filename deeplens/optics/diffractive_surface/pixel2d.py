"""Pixel2D DOE parameterization. Each pixel is an independent parameter."""

import torch
from deeplens.optics.diffractive_surface.diffractive import DiffractiveSurface


class Pixel2D(DiffractiveSurface):
    """Pixel2D DOE parameterization - direct phase map representation."""

    def __init__(
        self,
        d,
        phase_map_path=None,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
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
        super().__init__(d=d, res=res, mat=mat, fab_ps=fab_ps, wvln0=wvln0, device=device)

        # Initialize phase map with random values
        if phase_map_path is None:
            self.phase_map = torch.randn(self.res, device=self.device) * 1e-3
        elif isinstance(phase_map_path, str):
            self.phase_map = torch.load(phase_map_path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"Invalid phase_map_path: {phase_map_path}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Pixel2D DOE from a dict."""
        d = doe_dict["d"]
        res = doe_dict.get("res", (2000, 2000))
        fab_ps = doe_dict.get("fab_ps", 0.001)
        phase_map_path = doe_dict.get("phase_map_path", None)
        wvln0 = doe_dict.get("wvln0", 0.55)
        mat = doe_dict.get("mat", "fused_silica")
        return cls(
            d=d,
            res=res,
            mat=mat,
            fab_ps=fab_ps,
            phase_map_path=phase_map_path,
            wvln0=wvln0,
        )

    def _phase_map0(self):
        """Get the phase map at design wavelength."""
        return self.phase_map

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization."""
        self.phase_map.requires_grad = True
        optimizer_params = [{"params": [self.phase_map], "lr": lr}]
        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, phase_map_path):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["phase_map_path"] = phase_map_path
        torch.save(self.phase_map.clone().detach().cpu(), phase_map_path)
        return surf_dict

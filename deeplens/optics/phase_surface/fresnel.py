"""Fresnel phase on a plane surface."""

import torch

from deeplens.optics.phase_surface.phase import Phase


class FresnelPhase(Phase):
    """Fresnel phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        f0=100.0,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=True,
        device="cpu",
    ):
        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # Focal length at 550nm
        self.f0 = torch.tensor(f0)
        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize Fresnel parameters."""
        self.param_model = "fresnel"

    @classmethod
    def init_from_dict(cls, param_dict):
        """Initialize FresnelPhase from a dictionary of parameters."""
        r = param_dict.get("r")
        d = param_dict.get("d")
        f0 = param_dict.get("f0", 100.0)
        norm_radii = param_dict.get("norm_radii", None)
        mat2 = param_dict.get("mat2", "air")
        pos_xy = param_dict.get("pos_xy", [0.0, 0.0])
        vec_local = param_dict.get("vec_local", [0.0, 0.0, 1.0])
        is_square = param_dict.get("is_square", True)
        device = param_dict.get("device", "cpu")
        return cls(
            r=r,
            d=d,
            f0=f0,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        phi = (
            -2 * torch.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1)
        )  # unit [mm]
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        dphidx = -2 * torch.pi * x / (0.55e-3 * self.f0)  # unit [mm]
        dphidy = -2 * torch.pi * y / (0.55e-3 * self.f0)
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Generate optimizer parameters."""
        params = []

        # Optimize focal length
        self.f0.requires_grad = True
        params.append({"params": [self.f0], "lr": lrs[0]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./fresnel_doe.pth"):
        """Save Fresnel DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "f0": self.f0.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./fresnel_doe.pth"):
        """Load Fresnel DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.f0 = ckpt["f0"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "f0": self.f0.item(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

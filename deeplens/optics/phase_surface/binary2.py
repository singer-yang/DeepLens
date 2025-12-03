"""Binary2 phase on a plane surface."""

import torch

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class Binary2Phase(Phase):
    """Binary2 phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        order2=0.0,
        order4=0.0,
        order6=0.0,
        order8=0.0,
        order10=0.0,
        order12=0.0,
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

        # Initialize polynomial coefficients
        self.order2 = torch.tensor(order2)
        self.order4 = torch.tensor(order4)
        self.order6 = torch.tensor(order6)
        self.order8 = torch.tensor(order8)
        self.order10 = torch.tensor(order10)
        self.order12 = torch.tensor(order12)

        self.to(device)
        self.init_param_model()

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize Binary2 phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            surf_dict.get("order2", 0.0),
            surf_dict.get("order4", 0.0),
            surf_dict.get("order6", 0.0),
            surf_dict.get("order8", 0.0),
            surf_dict.get("order10", 0.0),
            surf_dict.get("order12", 0.0),
            norm_radii,
            mat2,
        )
        return obj

    def init_param_model(self):
        """Initialize Binary2 parameters."""
        self.param_model = "binary2"

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        phi = (
            self.order2 * r_norm**2
            + self.order4 * r_norm**4
            + self.order6 * r_norm**6
            + self.order8 * r_norm**8
            + self.order10 * r_norm**10
            + self.order12 * r_norm**12
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        dphidr = (
            2 * self.order2 * r_norm
            + 4 * self.order4 * r_norm**3
            + 6 * self.order6 * r_norm**5
            + 8 * self.order8 * r_norm**7
            + 10 * self.order10 * r_norm**9
            + 12 * self.order12 * r_norm**11
        )
        dphidx = dphidr * x_norm / r_norm / self.norm_radii
        dphidy = dphidr * y_norm / r_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters."""
        params = []

        # Optimize position
        self.d.requires_grad = True
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize polynomial coefficients
        self.order2.requires_grad = True
        params.append({"params": [self.order2], "lr": lrs[1]})

        self.order4.requires_grad = True
        params.append({"params": [self.order4], "lr": lrs[1]})

        self.order6.requires_grad = True
        params.append({"params": [self.order6], "lr": lrs[1]})

        self.order8.requires_grad = True
        params.append({"params": [self.order8], "lr": lrs[1]})

        self.order10.requires_grad = True
        params.append({"params": [self.order10], "lr": lrs[1]})

        self.order12.requires_grad = True
        params.append({"params": [self.order12], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./binary2_doe.pth"):
        """Save Binary2 DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order8": self.order8.clone().detach().cpu(),
                "order10": self.order10.clone().detach().cpu(),
                "order12": self.order12.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./binary2_doe.pth"):
        """Load Binary2 DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order8 = ckpt["order8"].to(self.device)
        self.order10 = ckpt["order10"].to(self.device)
        self.order12 = ckpt["order12"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": round(self.order2.item(), 4),
            "order4": round(self.order4.item(), 4),
            "order6": round(self.order6.item(), 4),
            "order8": round(self.order8.item(), 4),
            "order10": round(self.order10.item(), 4),
            "order12": round(self.order12.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

"""Polynomial phase on a plane surface."""

import numpy as np
import torch

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class PolyPhase(Phase):
    """Polynomial phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        order2=0.0,
        order3=0.0,
        order4=0.0,
        order5=0.0,
        order6=0.0,
        order7=0.0,
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

        # Initialize polynomial coefficients with random small values if not provided
        rand_value = np.random.rand(6) * 0.001
        self.order2 = torch.tensor(order2 if order2 != 0.0 else rand_value[0])
        self.order3 = torch.tensor(order3 if order3 != 0.0 else rand_value[1])
        self.order4 = torch.tensor(order4 if order4 != 0.0 else rand_value[2])
        self.order5 = torch.tensor(order5 if order5 != 0.0 else rand_value[3])
        self.order6 = torch.tensor(order6 if order6 != 0.0 else rand_value[4])
        self.order7 = torch.tensor(order7 if order7 != 0.0 else rand_value[5])

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize Poly1D parameters."""
        self.param_model = "poly1d"

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        phi_even = (
            self.order2 * r_norm**2 + self.order4 * r_norm**4 + self.order6 * r_norm**6
        )
        phi_odd = (
            self.order3 * (x_norm**3 + y_norm**3)
            + self.order5 * (x_norm**5 + y_norm**5)
            + self.order7 * (x_norm**7 + y_norm**7)
        )
        phi = phi_even + phi_odd

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        dphi_even_dr = (
            2 * self.order2 * r_norm
            + 4 * self.order4 * r_norm**3
            + 6 * self.order6 * r_norm**5
        )
        dphi_even_dx = dphi_even_dr * x_norm / r_norm / self.norm_radii
        dphi_even_dy = dphi_even_dr * y_norm / r_norm / self.norm_radii

        dphi_odd_dx = (
            3 * self.order3 * x_norm**2
            + 5 * self.order5 * x_norm**4
            + 7 * self.order7 * x_norm**6
        ) / self.norm_radii
        dphi_odd_dy = (
            3 * self.order3 * y_norm**2
            + 5 * self.order5 * y_norm**4
            + 7 * self.order7 * y_norm**6
        ) / self.norm_radii

        dphidx = dphi_even_dx + dphi_odd_dx
        dphidy = dphi_even_dy + dphi_odd_dy

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], optim_mat=False
    ):
        """Generate optimizer parameters."""
        params = []

        # Optimize polynomial coefficients with different learning rates
        self.order2.requires_grad = True
        self.order3.requires_grad = True
        self.order4.requires_grad = True
        self.order5.requires_grad = True
        self.order6.requires_grad = True
        self.order7.requires_grad = True

        params.append({"params": [self.order2], "lr": lrs[0]})
        params.append({"params": [self.order3], "lr": lrs[1]})
        params.append({"params": [self.order4], "lr": lrs[2]})
        params.append({"params": [self.order5], "lr": lrs[3]})
        params.append({"params": [self.order6], "lr": lrs[4]})
        params.append({"params": [self.order7], "lr": lrs[5]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./poly1d_doe.pth"):
        """Save Poly1D DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order3": self.order3.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order5": self.order5.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order7": self.order7.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./poly1d_doe.pth"):
        """Load Poly1D DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order3 = ckpt["order3"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order5 = ckpt["order5"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order7 = ckpt["order7"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": round(self.order2.item(), 4),
            "order3": round(self.order3.item(), 4),
            "order4": round(self.order4.item(), 4),
            "order5": round(self.order5.item(), 4),
            "order6": round(self.order6.item(), 4),
            "order7": round(self.order7.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

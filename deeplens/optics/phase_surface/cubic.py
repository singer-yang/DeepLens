"""Cubic phase on a plane surface."""

import numpy as np
import torch

from deeplens.optics.phase_surface.phase import Phase


class CubicPhase(Phase):
    """Cubic phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        coeff_x3=0.0,
        coeff_y3=0.0,
        coeff_x2y=0.0,
        coeff_xy2=0.0,
        coeff_x3y=0.0,
        coeff_xy3=0.0,
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

        # Initialize cubic polynomial coefficients with random small values if not provided
        rand_value = np.random.rand(6) * 0.001
        self.coeff_x3 = torch.tensor(coeff_x3 if coeff_x3 != 0.0 else rand_value[0])
        self.coeff_y3 = torch.tensor(coeff_y3 if coeff_y3 != 0.0 else rand_value[1])
        self.coeff_x2y = torch.tensor(coeff_x2y if coeff_x2y != 0.0 else rand_value[2])
        self.coeff_xy2 = torch.tensor(coeff_xy2 if coeff_xy2 != 0.0 else rand_value[3])
        self.coeff_x3y = torch.tensor(coeff_x3y if coeff_x3y != 0.0 else rand_value[4])
        self.coeff_xy3 = torch.tensor(coeff_xy3 if coeff_xy3 != 0.0 else rand_value[5])

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize cubic parameters."""
        self.param_model = "cubic"

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = (
            self.coeff_x3 * x_norm**3
            + self.coeff_y3 * y_norm**3
            + self.coeff_x2y * x_norm**2 * y_norm
            + self.coeff_xy2 * x_norm * y_norm**2
            + self.coeff_x3y * x_norm**3 * y_norm
            + self.coeff_xy3 * x_norm * y_norm**3
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # Derivatives with respect to normalized coordinates
        dphi_dx_norm = (
            3 * self.coeff_x3 * x_norm**2
            + 2 * self.coeff_x2y * x_norm * y_norm
            + 3 * self.coeff_x3y * x_norm**2 * y_norm
            + self.coeff_xy2 * y_norm**2
            + self.coeff_xy3 * y_norm**3
        )

        dphi_dy_norm = (
            3 * self.coeff_y3 * y_norm**2
            + self.coeff_x2y * x_norm**2
            + 2 * self.coeff_xy2 * x_norm * y_norm
            + self.coeff_x3y * x_norm**3
            + 3 * self.coeff_xy3 * x_norm * y_norm**2
        )

        # Convert back to physical coordinates
        dphidx = dphi_dx_norm / self.norm_radii
        dphidy = dphi_dy_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5], optim_mat=False
    ):
        """Generate optimizer parameters."""
        params = []

        # Optimize cubic polynomial coefficients with different learning rates
        self.coeff_x3.requires_grad = True
        self.coeff_y3.requires_grad = True
        self.coeff_x2y.requires_grad = True
        self.coeff_xy2.requires_grad = True
        self.coeff_x3y.requires_grad = True
        self.coeff_xy3.requires_grad = True

        params.append({"params": [self.coeff_x3], "lr": lrs[0]})
        params.append({"params": [self.coeff_y3], "lr": lrs[1]})
        params.append({"params": [self.coeff_x2y], "lr": lrs[2]})
        params.append({"params": [self.coeff_xy2], "lr": lrs[3]})
        params.append({"params": [self.coeff_x3y], "lr": lrs[4]})
        params.append({"params": [self.coeff_xy3], "lr": lrs[5]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./cubic_doe.pth"):
        """Save cubic DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "coeff_x3": self.coeff_x3.clone().detach().cpu(),
                "coeff_y3": self.coeff_y3.clone().detach().cpu(),
                "coeff_x2y": self.coeff_x2y.clone().detach().cpu(),
                "coeff_xy2": self.coeff_xy2.clone().detach().cpu(),
                "coeff_x3y": self.coeff_x3y.clone().detach().cpu(),
                "coeff_xy3": self.coeff_xy3.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./cubic_doe.pth"):
        """Load cubic DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.coeff_x3 = ckpt["coeff_x3"].to(self.device)
        self.coeff_y3 = ckpt["coeff_y3"].to(self.device)
        self.coeff_x2y = ckpt["coeff_x2y"].to(self.device)
        self.coeff_xy2 = ckpt["coeff_xy2"].to(self.device)
        self.coeff_x3y = ckpt["coeff_x3y"].to(self.device)
        self.coeff_xy3 = ckpt["coeff_xy3"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "coeff_x3": round(self.coeff_x3.item(), 4),
            "coeff_y3": round(self.coeff_y3.item(), 4),
            "coeff_x2y": round(self.coeff_x2y.item(), 4),
            "coeff_xy2": round(self.coeff_xy2.item(), 4),
            "coeff_x3y": round(self.coeff_x3y.item(), 4),
            "coeff_xy3": round(self.coeff_xy3.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

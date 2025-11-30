"""Quartic (Q-type) phase on a plane surface."""

import numpy as np
import torch

from deeplens.optics.phase_surface.phase import Phase


class QuarticPhase(Phase):
    """Quartic phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        coeff_x4=0.0,
        coeff_y4=0.0,
        coeff_x3y=0.0,
        coeff_xy3=0.0,
        coeff_x2y2=0.0,
        coeff_x4y=0.0,
        coeff_xy4=0.0,
        coeff_x3y2=0.0,
        coeff_x2y3=0.0,
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

        # Initialize quartic polynomial coefficients with random small values if not provided
        rand_value = np.random.rand(9) * 0.001
        self.coeff_x4 = torch.tensor(coeff_x4 if coeff_x4 != 0.0 else rand_value[0])
        self.coeff_y4 = torch.tensor(coeff_y4 if coeff_y4 != 0.0 else rand_value[1])
        self.coeff_x3y = torch.tensor(coeff_x3y if coeff_x3y != 0.0 else rand_value[2])
        self.coeff_xy3 = torch.tensor(coeff_xy3 if coeff_xy3 != 0.0 else rand_value[3])
        self.coeff_x2y2 = torch.tensor(coeff_x2y2 if coeff_x2y2 != 0.0 else rand_value[4])
        self.coeff_x4y = torch.tensor(coeff_x4y if coeff_x4y != 0.0 else rand_value[5])
        self.coeff_xy4 = torch.tensor(coeff_xy4 if coeff_xy4 != 0.0 else rand_value[6])
        self.coeff_x3y2 = torch.tensor(coeff_x3y2 if coeff_x3y2 != 0.0 else rand_value[7])
        self.coeff_x2y3 = torch.tensor(coeff_x2y3 if coeff_x2y3 != 0.0 else rand_value[8])

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize quartic parameters."""
        self.param_model = "quartic"

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = (
            self.coeff_x4 * x_norm**4
            + self.coeff_y4 * y_norm**4
            + self.coeff_x3y * x_norm**3 * y_norm
            + self.coeff_xy3 * x_norm * y_norm**3
            + self.coeff_x2y2 * x_norm**2 * y_norm**2
            + self.coeff_x4y * x_norm**4 * y_norm
            + self.coeff_xy4 * x_norm * y_norm**4
            + self.coeff_x3y2 * x_norm**3 * y_norm**2
            + self.coeff_x2y3 * x_norm**2 * y_norm**3
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # Derivatives with respect to normalized coordinates
        dphi_dx_norm = (
            4 * self.coeff_x4 * x_norm**3
            + 3 * self.coeff_x3y * x_norm**2 * y_norm
            + self.coeff_xy3 * y_norm**3
            + 2 * self.coeff_x2y2 * x_norm * y_norm**2
            + 4 * self.coeff_x4y * x_norm**3 * y_norm
            + self.coeff_xy4 * y_norm**4
            + 3 * self.coeff_x3y2 * x_norm**2 * y_norm**2
            + 2 * self.coeff_x2y3 * x_norm * y_norm**3
        )

        dphi_dy_norm = (
            4 * self.coeff_y4 * y_norm**3
            + self.coeff_x3y * x_norm**3
            + 3 * self.coeff_xy3 * x_norm * y_norm**2
            + 2 * self.coeff_x2y2 * x_norm**2 * y_norm
            + self.coeff_x4y * x_norm**4
            + 4 * self.coeff_xy4 * x_norm * y_norm**3
            + 2 * self.coeff_x3y2 * x_norm**3 * y_norm
            + 3 * self.coeff_x2y3 * x_norm**2 * y_norm**2
        )

        # Convert back to physical coordinates
        dphidx = dphi_dx_norm / self.norm_radii
        dphidy = dphi_dy_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5], optim_mat=False
    ):
        """Generate optimizer parameters."""
        params = []

        # Optimize quartic polynomial coefficients with different learning rates
        self.coeff_x4.requires_grad = True
        self.coeff_y4.requires_grad = True
        self.coeff_x3y.requires_grad = True
        self.coeff_xy3.requires_grad = True
        self.coeff_x2y2.requires_grad = True
        self.coeff_x4y.requires_grad = True
        self.coeff_xy4.requires_grad = True
        self.coeff_x3y2.requires_grad = True
        self.coeff_x2y3.requires_grad = True

        params.append({"params": [self.coeff_x4], "lr": lrs[0]})
        params.append({"params": [self.coeff_y4], "lr": lrs[1]})
        params.append({"params": [self.coeff_x3y], "lr": lrs[2]})
        params.append({"params": [self.coeff_xy3], "lr": lrs[3]})
        params.append({"params": [self.coeff_x2y2], "lr": lrs[4]})
        params.append({"params": [self.coeff_x4y], "lr": lrs[5]})
        params.append({"params": [self.coeff_xy4], "lr": lrs[6]})
        params.append({"params": [self.coeff_x3y2], "lr": lrs[7]})
        params.append({"params": [self.coeff_x2y3], "lr": lrs[8]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./quartic_doe.pth"):
        """Save quartic DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "coeff_x4": self.coeff_x4.clone().detach().cpu(),
                "coeff_y4": self.coeff_y4.clone().detach().cpu(),
                "coeff_x3y": self.coeff_x3y.clone().detach().cpu(),
                "coeff_xy3": self.coeff_xy3.clone().detach().cpu(),
                "coeff_x2y2": self.coeff_x2y2.clone().detach().cpu(),
                "coeff_x4y": self.coeff_x4y.clone().detach().cpu(),
                "coeff_xy4": self.coeff_xy4.clone().detach().cpu(),
                "coeff_x3y2": self.coeff_x3y2.clone().detach().cpu(),
                "coeff_x2y3": self.coeff_x2y3.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./quartic_doe.pth"):
        """Load quartic DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.coeff_x4 = ckpt["coeff_x4"].to(self.device)
        self.coeff_y4 = ckpt["coeff_y4"].to(self.device)
        self.coeff_x3y = ckpt["coeff_x3y"].to(self.device)
        self.coeff_xy3 = ckpt["coeff_xy3"].to(self.device)
        self.coeff_x2y2 = ckpt["coeff_x2y2"].to(self.device)
        self.coeff_x4y = ckpt["coeff_x4y"].to(self.device)
        self.coeff_xy4 = ckpt["coeff_xy4"].to(self.device)
        self.coeff_x3y2 = ckpt["coeff_x3y2"].to(self.device)
        self.coeff_x2y3 = ckpt["coeff_x2y3"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "coeff_x4": round(self.coeff_x4.item(), 4),
            "coeff_y4": round(self.coeff_y4.item(), 4),
            "coeff_x3y": round(self.coeff_x3y.item(), 4),
            "coeff_xy3": round(self.coeff_xy3.item(), 4),
            "coeff_x2y2": round(self.coeff_x2y2.item(), 4),
            "coeff_x4y": round(self.coeff_x4y.item(), 4),
            "coeff_xy4": round(self.coeff_xy4.item(), 4),
            "coeff_x3y2": round(self.coeff_x3y2.item(), 4),
            "coeff_x2y3": round(self.coeff_x2y3.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

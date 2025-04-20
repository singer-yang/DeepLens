"""Spheric surface."""

import numpy as np
import torch

from .base import EPSILON, Surface


class Spheric(Surface):
    def __init__(self, c, r, d, mat2, device="cpu"):
        super(Spheric, self).__init__(r, d, mat2, is_square=False, device=device)
        self.c = torch.tensor(c)

        self.c_perturb = 0.0
        self.d_perturb = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]
        return cls(c, surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surfaces sag z = r**2 * c / (1 - sqrt(1 - r**2 * c**2))"""
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt(1 - r2 * c**2))
        return sag

    def _dfdxy(self, x, y):
        """Compute surface sag derivatives to x and y: dz / dx, dz / dy."""
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)
        dfdr2 = c / (2 * sf)

        dfdx = dfdr2 * 2 * x
        dfdy = dfdr2 * 2 * y

        return dfdx, dfdy

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of the surface sag z = sag(x, y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Returns:
            d2f_dx2 (tensor): ∂²f / ∂x²
            d2f_dxdy (tensor): ∂²f / ∂x∂y
            d2f_dy2 (tensor): ∂²f / ∂y²
        """
        c = self.c + self.c_perturb
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)

        # First derivative (df/dr2)
        dfdr2 = c / (2 * sf)

        # Second derivative (d²f/dr2²)
        d2f_dr2_dr2 = (c**3) / (4 * sf**3)

        # Compute second-order partial derivatives using the chain rule
        d2f_dx2 = 4 * x**2 * d2f_dr2_dr2 + 2 * dfdr2
        d2f_dxdy = 4 * x * y * d2f_dr2_dr2
        d2f_dy2 = 4 * y**2 * d2f_dr2_dr2 + 2 * dfdr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        c = self.c + self.c_perturb

        valid = (x**2 + y**2) < 1 / c**2
        return valid

    def max_height(self):
        """Maximum valid height."""
        c = self.c + self.c_perturb

        max_height = torch.sqrt(1 / c**2).item() - 0.01
        return max_height

    # =========================================
    # Manufacturing
    # =========================================
    def perturb(self, tolerance):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        self.r_offset = np.random.randn() * tolerance.get("r", 0.001)
        self.d_offset = np.random.randn() * tolerance.get("d", 0.001)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lr=[0.001, 0.001], optim_mat=False):
        """Activate gradient computation for c and d and return optimizer parameters."""
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.c], "lr": lr[0]})
        params.append({"params": [self.d], "lr": lr[1]})

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        roc = 1 / self.c.item() if self.c.item() != 0 else 0.0
        surf_dict = {
            "type": "Spheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(roc, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r} 1 0 0 1 ""
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str

"""Cylindric surface.

Usage ref:
[1] https://images.app.goo.gl/Yzrx7De6Hy1mSb18A
[2] https://images.app.goo.gl/LFsS6kHu28wE8yot8
[3] https://images.app.goo.gl/XvgP3fJBpUEejxc16
[4] https://images.app.goo.gl/KXDL5Eb4UD61nCEF8
"""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


class Cylindric(Surface):
    def __init__(self, c, r, d, mat2, device="cpu"):
        super(Cylindric, self).__init__(r, d, mat2, is_square=False, device=device)
        self.c = torch.tensor(c)

        self.c_perturb = 0.0
        self.d_perturb = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict and surf_dict["roc"] != 0:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]
        return cls(c, surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surfaces sag z = c * x**2 / (1 + sqrt(1 - c**2 * x**2))
        
        The sag is defined along x direction, which means the cylindar is placed along y-axis.
        """
        c = self.c + self.c_perturb

        x2 = x**2
        # Add a small epsilon to prevent sqrt of negative number
        sf = torch.sqrt(1 - x2 * c**2 + 1e-8)
        sag = c * x2 / (1 + sf)
        return sag

    def _dfdxy(self, x, y):
        """Compute surface sag derivatives to x and y: dz / dx, dz / dy."""
        c = self.c + self.c_perturb

        x2 = x**2
        sf = torch.sqrt(1 - x2 * c**2 + EPSILON)

        # dz/dx = c*x / sqrt(1 - c^2 * x^2)
        dfdx = c * x / sf
        dfdy = torch.zeros_like(y)

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
        x2 = x**2
        sf = torch.sqrt(1 - x2 * c**2 + EPSILON)

        # d2z/dx2 = c / (1 - c^2 * x^2)^(3/2)
        d2f_dx2 = c / (sf**3)
        d2f_dxdy = torch.zeros_like(x)
        d2f_dy2 = torch.zeros_like(y)

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        c = self.c + self.c_perturb

        # The sag is defined only where the argument of the square root is non-negative
        if c != 0:
            valid = (x**2) < 1 / c**2
        else:
            valid = torch.ones_like(x, dtype=torch.bool)
        return valid

    def max_height(self):
        """Maximum valid height."""
        c = self.c + self.c_perturb

        if c != 0:
            max_height = torch.sqrt(1 / c**2).item() - 0.01
        else:
            max_height = self.r # For a flat surface, max_height is radius
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
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """Activate gradient computation for c and d and return optimizer parameters."""
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})
        params.append({"params": [self.c], "lr": lrs[1]})

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
            "type": "Cylindric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(roc, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        raise NotImplementedError(
            "zmx_str() is not implemented for {}".format(self.__class__.__name__)
        )

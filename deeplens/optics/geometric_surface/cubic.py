"""Cubic surface."""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import Surface


class Cubic(Surface):
    def __init__(
        self,
        r,
        d,
        b,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.b = torch.tensor(b)

        if len(b) == 1:
            self.b3 = torch.tensor(b[0])
            self.b_degree = 1
        elif len(b) == 2:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b_degree = 2
        elif len(b) == 3:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b7 = torch.tensor(b[2])
            self.b_degree = 3
        else:
            raise ValueError("Unsupported cubic degree!")

        self.rotate_angle = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["b"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surface height z(x, y)."""
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = (
                self.b3 * (x**3 + y**3)
                + self.b5 * (x**5 + y**5)
                + self.b7 * (x**7 + y**7)
            )
        else:
            raise ValueError("Unsupported cubic degree!")

        if len(z.size()) == 0:
            z = torch.tensor(z).to(self.device)

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return z

    def _dfdxy(self, x, y):
        """Compute surface height derivatives to x and y."""
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise ValueError("Unsupported cubic degree!")

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return sx, sy

    def get_optimizer_params(self, lrs=[1e-4], decay=0.1, optim_mat=False):
        """Return parameters for optimizer."""
        # Broadcast learning rates to all cubic coefficients
        if len(lrs) == 1:
            lrs = lrs + [
                lrs[0] * decay ** (b_degree + 1)
                for b_degree in range(self.b_degree - 1)
            ]

        params = []

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize cubic coefficients
        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
            params.append({"params": [self.b7], "lr": lrs[3]})
        else:
            raise ValueError("Unsupported cubic degree!")

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # Manufacturing
    # =========================================
    def perturb(self, tolerance_params):
        """Perturb the surface"""
        super().perturb(tolerance_params)
        self.r_error = np.random.randn() * tolerance_params.get("r_tole", 0.001)
        if self.d.item() != 0:
            self.d_error = np.random.randn() * tolerance_params.get("d_tole", 0.0005)

        if self.b_degree == 1:
            self.b3_error = np.random.randn() * tolerance_params.get("b3_tole", 0.001)
        elif self.b_degree == 2:
            self.b3_error = np.random.randn() * tolerance_params.get("b3_tole", 0.001)
            self.b5_error = np.random.randn() * tolerance_params.get("b5_tole", 0.001)
        elif self.b_degree == 3:
            self.b3_error = np.random.randn() * tolerance_params.get("b3_tole", 0.001)
            self.b5_error = np.random.randn() * tolerance_params.get("b5_tole", 0.001)
            self.b7_error = np.random.randn() * tolerance_params.get("b7_tole", 0.001)

        self.rotate_angle_error = np.random.randn() * tolerance_params.get(
            "angle_tole", 0.01
        )

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        return {
            "type": "Cubic",
            "b3": self.b3.item(),
            "b5": self.b5.item(),
            "b7": self.b7.item(),
            "r": self.r,
            "(d)": round(self.d.item(), 4),
        }

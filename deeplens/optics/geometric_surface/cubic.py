"""Cubic surface.

Typical equation: z(x,y) = b3 * (x^3 + y^3)
"""
import numpy as np
import torch

from .base import Surface


class Cubic(Surface):
    def __init__(self, r, d, b, mat2, is_square=False, device="cpu"):
        Surface.__init__(self, r, d, mat2, is_square=is_square, device=device)
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

    def get_optimizer_params(self, lr, optim_mat=False):
        """Return parameters for optimizer."""
        params = []

        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lr})

        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
            params.append({"params": [self.b5], "lr": lr * 0.1})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
            params.append({"params": [self.b5], "lr": lr * 0.1})
            params.append({"params": [self.b7], "lr": lr * 0.01})
        else:
            raise ValueError("Unsupported cubic degree!")

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    
    # =========================================
    # Manufacturing
    # =========================================
    def perturb(self, tolerance):
        """Perturb the surface"""
        self.r_offset = np.random.randn() * tolerance.get("r", 0.001)
        if self.d != 0:
            self.d_offset = np.random.randn() * tolerance.get("d", 0.0005)

        if self.b_degree == 1:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
        elif self.b_degree == 2:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
            self.b5_offset = np.random.randn() * tolerance.get("b5", 0.001)
        elif self.b_degree == 3:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
            self.b5_offset = np.random.randn() * tolerance.get("b5", 0.001)
            self.b7_offset = np.random.randn() * tolerance.get("b7", 0.001)

        self.rotate_angle = np.random.randn() * tolerance.get("angle", 0.01)


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

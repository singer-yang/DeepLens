"""Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base."""

import numpy as np
import torch

from .base import Surface


class Plane(Surface):
    def __init__(self, r, d, mat2, is_square=False, device="cpu"):
        """Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base."""
        Surface.__init__(self, r, d, mat2=mat2, is_square=is_square, device=device)
        self.l = r * np.sqrt(2)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection and update ray data."""
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.ra > 0)
            )
        else:
            valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
                ray.ra > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)

        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points."""
        n = torch.zeros_like(ray.d)
        n[..., 2] = -1
        return n

    def _sag(self, x, y):
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def _d2fdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        surf_dict = {
            "type": "Plane",
            "(l)": self.l,
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "is_square": True,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

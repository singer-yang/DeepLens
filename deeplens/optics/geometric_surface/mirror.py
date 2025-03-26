"""Mirror surface."""
import numpy as np
import torch

from .base import Surface


class Mirror(Surface):
    def __init__(self, l, d, device="cpu"):
        """Mirror surface."""
        Surface.__init__(
            self, l / np.sqrt(2), d, mat2="air", is_square=True, device=device
        )
        self.l = l

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["l"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, **kwargs):
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (
            (torch.abs(new_o[..., 0]) < self.w / 2)
            & (torch.abs(new_o[..., 1]) < self.h / 2)
            & (ray.ra > 0)
        )

        # Update ray position
        new_o = ray.o + ray.d * t.unsqueeze(-1)

        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + 1.0 * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def ray_reaction(self, ray, **kwargs):
        """Compute output ray after intersection and refraction with the mirror surface."""
        # Intersection
        ray = self.intersect(ray)

        # Reflection
        ray = self.reflect(ray)

        return ray

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "l": self.l,
            "d": self.d,
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

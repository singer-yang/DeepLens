"""Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base."""

import torch

from deeplens.optics.geometric_surface.base import Surface


class Plane(Surface):
    def __init__(
        self,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Plane surface.

        Examples:
            - IR filter.
            - Lens cover glass.
            - DOE base.

        The following surfaces inherit from Plane:
            - Aperture.
            - Mirror.
            - ThinLens.
        """
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

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection in local coordinate system and update ray data."""
        # Solve intersection
        t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        
        # Aperture mask
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.is_valid > 0)
            )
        else:
            valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
                ray.is_valid > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.coherent:
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points in local coordinate system.

        Normal vector points from the surface toward the side where the light is coming from.
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def _sag(self, x, y):
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def _d2fdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Activate gradient computation for d and return optimizer parameters."""
        params = []

        # Optimize d
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        surf_dict = {
            "type": "Plane",
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

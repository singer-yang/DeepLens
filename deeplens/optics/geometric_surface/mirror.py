"""Mirror surface."""
import numpy as np
import torch

from deeplens.optics.geometric_surface.base import Surface


class Mirror(Surface):
    def __init__(self, r, d, surf_idx=None, origin=None, vec_local=[0., 0., 1.], mat2=None, is_square=True, device="cpu"):
        """Mirror surface."""
        Surface.__init__(self, r=r, d=d, mat2="air", is_square=is_square, origin=origin, vec_local=vec_local, device=device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        surf_idx = surf_dict.get("surf_idx", None)
        return cls(surf_dict["r"], surf_dict["d"], surf_idx=surf_idx)

    def intersect(self, ray, n=None):
        """Solve ray-surface intersection and update ray data."""
        w, h = self.w, self.h
        
        # Solve intersection
        # t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        t = (0. - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (
            (torch.abs(new_o[..., 0]) < w / 2)
            & (torch.abs(new_o[..., 1]) < h / 2)
            & (ray.valid > 0)
        )
    
        # Update ray position
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.valid = ray.valid * valid

        if ray.coherent:
            ray.opl = torch.where(valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl)

        return ray

    def ray_reaction(self, ray, n1=None, n2=None):
        """Compute output ray after intersection and reflection with the mirror surface."""
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.reflect(ray)
        ray = self.to_global_coord(ray)
        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at the intersection point in local coordinate system."""
        n_vec = torch.tensor([0., 0., 1.], device=ray.device)
        return n_vec

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "surf_idx": self.surf_idx,
            "type": self.__class__.__name__,
            "r": self.r,
            "d": self.d,
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

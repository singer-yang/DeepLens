"""Mirror surface."""

from deeplens.optics.geometric_surface.base import Surface
from deeplens.optics.geometric_surface.plane import Plane


class Mirror(Plane):
    def __init__(
        self,
        r,
        d,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        mat2=None,
        is_square=True,
        device="cpu",
    ):
        """Mirror surface."""
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2="air",
            is_square=is_square,
            pos_xy=pos_xy,
            vec_local=vec_local,
            device=device,
        )

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"])

    def ray_reaction(self, ray, n1=None, n2=None):
        """Compute output ray after intersection and reflection with the mirror surface."""
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.reflect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "d": self.d,
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

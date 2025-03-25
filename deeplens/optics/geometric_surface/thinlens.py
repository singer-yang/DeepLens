"""Thin lens element. Both sides are in air."""

import torch
import torch.nn.functional as F

from .base import Surface


class ThinLens(Surface):
    def __init__(self, f, r, d, mat2="air", is_square=False, device="cpu"):
        """Thin lens surface."""
        Surface.__init__(self, r, d, mat2=mat2, is_square=is_square, device=device)
        self.f = torch.tensor(f)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["f"], surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection and update rays."""
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
            ray.ra > 0
        )

        # Update ray position
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def refract(self, ray, n=1.0):
        """For a thin lens, all rays will converge to z = f plane. Therefore we trace the chief-ray (parallel-shift to surface center) to find the final convergence point for each ray.

        For coherent ray tracing, we can think it as a Fresnel lens with infinite refractive index.
        (1) Lens maker's equation
        (2) Spherical lens function
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0

        # Calculate convergence point
        if forward:
            t0 = self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(
                xy_final[..., 0].unsqueeze(-1), self.d.item() + self.f.item()
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)
        else:
            t0 = -self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(
                xy_final[..., 0].unsqueeze(-1), self.d.item() - self.f.item()
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)

        # New ray direction
        new_d = o_final - ray.o
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = new_d

        # Optical path length change
        if ray.coherent:
            if forward:
                ray.opl = (
                    ray.opl
                    - (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
            else:
                ray.opl = (
                    ray.opl
                    + (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )

        return ray

    def _sag(self, x, y):
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # Visualization
    # =========================================
    def draw_widget(self, ax, color="black", linestyle="-"):
        d = self.d.item()
        r = self.r
        ax.annotate(
            "",
            xy=(d, r),
            xytext=(d, -r),
            arrowprops=dict(
                arrowstyle="<->", color=color, linestyle=linestyle, linewidth=0.75
            ),
        )

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        surf_dict = {
            "type": "ThinLens",
            "f": round(self.f.item(), 4),
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
        }

        return surf_dict

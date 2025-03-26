"""Aperture surface."""

import numpy as np
import torch

from .base import Surface


class Aperture(Surface):
    def __init__(self, r, d, diffraction=False, device="cpu"):
        """Aperture surface."""
        Surface.__init__(self, r, d, mat2="air", is_square=False, device=device)
        self.diffraction = diffraction
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "diffraction" in surf_dict:
            diffraction = surf_dict["diffraction"]
        else:
            diffraction = False
        return cls(surf_dict["r"], surf_dict["d"], diffraction)

    def ray_reaction(self, ray, n1=1.0, n2=1.0, refraction=False):
        """Compute output ray after intersection and refraction."""
        # Intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) <= self.r) & (
            ray.ra > 0
        )

        # Update position
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        # Update phase
        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        # Diffraction
        if self.diffraction:
            raise Exception("Diffraction is not implemented for aperture.")

        return ray

    def _sag(self, x, y):
        """Compute surface height (always zero for aperture)."""
        return torch.zeros_like(x)

    # =======================================
    # Visualization
    # =======================================
    def draw_widget(self, ax, color="orange", linestyle="solid"):
        """Draw aperture wedge on the figure."""
        d = self.d.item()
        aper_wedge_l = 0.05 * self.r  # [mm]
        aper_wedge_h = 0.15 * self.r  # [mm]

        # Parallel edges
        z = np.linspace(d - aper_wedge_l, d + aper_wedge_l, 3)
        x = -self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

        # Vertical edges
        z = d * np.ones(3)
        x = np.linspace(self.r, self.r + aper_wedge_h, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = np.linspace(-self.r - aper_wedge_h, -self.r, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

    def draw_widget3D(self, ax, color="black"):
        """Draw the aperture as a circle in a 3D plot."""
        # Draw the edge circle
        theta = np.linspace(0, 2 * np.pi, 100)
        edge_x = self.r * np.cos(theta)
        edge_y = self.r * np.sin(theta)
        edge_z = np.full_like(edge_x, self.d.item())  # Constant z at aperture position

        # Plot the edge circle
        line = ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.5)

        return line

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Dict of surface parameters."""
        surf_dict = {
            "type": "Aperture",
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
            "is_square": self.is_square,
            "diffraction": self.diffraction,
        }
        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Zemax surface string."""
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
"""
        return zmx_str

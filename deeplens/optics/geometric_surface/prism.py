"""Prism surface consisting of entry plane, mirror, and exit plane in sequential mode."""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import Surface
from deeplens.optics.geometric_surface.plane import Plane  
from deeplens.optics.geometric_surface.mirror import Mirror


class Prism(Surface):
    def __init__(self, r, d, mirror_angle=45.0, mat2="air", device="cpu"):
        """Prism surface with entry plane, mirror, and exit plane in sequential mode.

        Prism local coordinate is defined the same as the first plane.
        
        Args:
            r (float): Aperture radius
            d (float): Distance to prism entry plane
            mirror_angle (float): Mirror angle in degrees (default: 45.0)
            mat2 (str): Material after prism (default: "air")
            device (str): Device for computations (default: "cpu")
        """
        Surface.__init__(self, r, d, mat2=mat2, is_square=True, device=device)
        
        self.mirror_angle = torch.tensor(mirror_angle * np.pi / 180.0)
        self._init_surfaces()
        
    def _init_surfaces(self):
        """Initialize the three surfaces: entry plane, mirror, exit plane.
        
        Current prism shape:
                               ^ ray out
                               |
                            _______
                            |    /
              ray in    ->  |  /
                            |/
        """
        d = self.d.item()
        mat2 = self.mat2.get_name()
        r = self.r
        device = self.device
        mirror_angle = self.mirror_angle.item()
        
        # Entry plane at the prism entrance
        plane1_d = d
        origin = [0., 0., d]
        vec_local = [0., 0., 1.]
        self.plane1 = Plane(r=r, origin=origin, vec_local=vec_local, d=plane1_d, mat2=mat2, device=device)
        
        # Mirror positioned inside the prism 
        mirror_d = r * float(np.tan(mirror_angle))
        origin = [0., 0., d + mirror_d]
        vec_local = [0., -1., 1.]
        self.mirror = Mirror(r=r, origin=origin, vec_local=vec_local, d=mirror_d, mat2="air", device=device)
        
        # Exit plane at the prism exit
        plane2_d = d + mirror_d
        origin = [0., r, d + mirror_d]
        vec_local = [0., 1., 0.]
        self.exit_plane = Plane(r=r, origin=origin, vec_local=vec_local, d=plane2_d, mat2=mat2, device=device)

        self.surfaces = [self.plane1, self.mirror, self.exit_plane]
    
    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize prism from dictionary."""
        return cls(
            surf_dict["r"], 
            surf_dict["d"], 
            surf_dict.get("mirror_angle", 45.0),
            surf_dict.get("prism_height", 10.0),
            surf_dict.get("mat2", "air")
        )

    def ray_reaction(self, ray, n1, n2, refraction=True):
        """Compute output ray after sequential interaction with all three surfaces.
        
        This method traces rays through:
        1. Entry plane (intersection only)
        2. Mirror (intersection + reflection)  
        3. Exit plane (intersection only)
        """
        for surface in self.surfaces:
            ray = surface.ray_reaction(ray, n1, n2)
        return ray

    def intersect(self, ray, n=1.0):
        """Primary intersection method - delegates to ray_reaction for sequential processing."""
        return self.ray_reaction(ray, n, n, refraction=False)

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points.
        
        For a prism, this is primarily determined by the exit plane.
        """
        return self.exit_plane.normal_vec(ray)

    def _sag(self, x, y):
        """Surface sag - prism entry is flat."""
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """Surface derivatives - prism entry is flat."""
        return torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """Activate gradient computation and return optimizer parameters."""
        params = []
        
        # Distance parameter
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})
        
        # Mirror angle parameter
        self.mirror_angle.requires_grad_(True)
        params.append({"params": [self.mirror_angle], "lr": lrs[1]})
        
        return params

    # # =========================================
    # # Visualization
    # # =========================================
    # def draw_widget(self, ax, color="black", linestyle="-"):
    #     """Draw prism representation in 2D layout."""
    #     d = self.d.item()
    #     r = self.r
    #     h = self.prism_height
        
    #     # Draw prism outline
    #     # Entry plane
    #     ax.plot([d, d], [-r, r], color=color, linestyle=linestyle, linewidth=1.0)
        
    #     # Top and bottom of prism
    #     ax.plot([d, d + h], [r, r], color=color, linestyle=linestyle, linewidth=0.75)
    #     ax.plot([d, d + h], [-r, -r], color=color, linestyle=linestyle, linewidth=0.75)
        
    #     # Mirror (angled line inside prism)
    #     mirror_pos = d + h/2
    #     mirror_len = r * 1.5
    #     angle = self.mirror_angle.item()
        
    #     # Mirror endpoints
    #     x1 = mirror_pos - mirror_len/2 * np.cos(angle)
    #     y1 = -mirror_len/2 * np.sin(angle)
    #     x2 = mirror_pos + mirror_len/2 * np.cos(angle)  
    #     y2 = mirror_len/2 * np.sin(angle)
        
    #     ax.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle, linewidth=1.5)
        
    #     # Exit plane
    #     ax.plot([d + h, d + h], [-r, r], color=color, linestyle=linestyle, linewidth=1.0)

    # # =========================================
    # # IO
    # # =========================================
    # def surf_dict(self):
    #     """Return surface parameters as dictionary."""
    #     surf_dict = {
    #         "type": "Prism",
    #         "r": round(self.r, 4),
    #         "d": round(self.d.item(), 4),
    #         "mirror_angle": round(self.mirror_angle.item() * 180.0 / np.pi, 4),
    #         "prism_height": round(self.prism_height, 4),
    #         "mat2": self.mat2.get_name(),
    #     }
    #     return surf_dict

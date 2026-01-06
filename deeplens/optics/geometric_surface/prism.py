# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

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
        
        self.mirror_angle = torch.tensor(mirror_angle * torch.pi / 180.0)
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
        
        # Plane 1 at the prism entrance
        plane1_d = d
        pos_xy = [0., 0.]
        vec_local = [0., 0., 1.]
        self.plane1 = Plane(r=r, d=plane1_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)
        
        # Mirror inside the prism 
        mirror_d = d + r * float(np.tan(mirror_angle))
        pos_xy = [0., 0.]
        vec_local = [0., -1., 1.]
        self.mirror = Mirror(r=r, d=mirror_d, pos_xy=pos_xy, vec_local=vec_local, device=device)
        
        # Plane 2 at the prism exit
        plane2_d = mirror_d
        pos_xy = [0., r]
        vec_local = [0., 1., 0.]
        self.exit_plane = Plane(r=r, d=plane2_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)

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
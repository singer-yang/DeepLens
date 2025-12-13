# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Spheric surface."""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


class Spheric(Surface):
    def __init__(
        self,
        c,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        super(Spheric, self).__init__(
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.c = torch.tensor(c)

        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        return cls(
            c=c,
            r=surf_dict["r"],
            d=surf_dict["d"],
            mat2=surf_dict["mat2"],
        )

    def _sag(self, x, y):
        """Compute surfaces sag z = r**2 * c / (1 - sqrt(1 - r**2 * c**2))"""
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        # Compute surface sag
        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt(1 - r2 * c**2))
        return sag

    def _dfdxy(self, x, y):
        """Compute surface sag derivatives to x and y: dz / dx, dz / dy."""
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        # Compute surface sag derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)
        dfdr2 = c / (2 * sf)

        dfdx = dfdr2 * 2 * x
        dfdy = dfdr2 * 2 * y

        return dfdx, dfdy

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of the surface sag z = sag(x, y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Returns:
            d2f_dx2 (tensor): ∂²f / ∂x²
            d2f_dxdy (tensor): ∂²f / ∂x∂y
            d2f_dy2 (tensor): ∂²f / ∂y²
        """
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        # Compute surface sag derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)

        # First derivative (df/dr2)
        dfdr2 = c / (2 * sf)

        # Second derivative (d²f/dr2²)
        d2f_dr2_dr2 = (c**3) / (4 * sf**3)

        # Compute second-order partial derivatives using the chain rule
        d2f_dx2 = 4 * x**2 * d2f_dr2_dr2 + 2 * dfdr2
        d2f_dxdy = 4 * x * y * d2f_dr2_dr2
        d2f_dy2 = 4 * y**2 * d2f_dr2_dr2 + 2 * dfdr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection in local coordinate system using analytical method.

        Sphere equation: (x)^2 + (y)^2 + (z - R)^2 = R^2, where R = 1/c
        Ray equation: p(t) = o + t*d
        Solve quadratic equation for intersection parameter t.

        Args:
            ray (Ray): input ray.
            n (float, optional): refractive index. Defaults to 1.0.

        Returns:
            ray (Ray): ray with updated position and opl.
        """
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        if torch.abs(c) < EPSILON:
            # Handle flat surface as a plane
            t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
            new_o = ray.o + t.unsqueeze(-1) * ray.d
            valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
                ray.is_valid > 0
            )
        else:
            R = 1.0 / c

            # Vector from ray origin to sphere center at (0, 0, R)
            oc = ray.o.clone()
            oc[..., 2] = oc[..., 2] - R

            # Quadratic equation: a*t^2 + b*t + c = 0
            # a = d·d = 1 (since ray direction is normalized)
            # b = 2*(o-center)·d
            # c = (o-center)·(o-center) - R^2

            a = torch.sum(ray.d * ray.d, dim=-1)  # Should be 1 for normalized rays
            b = 2.0 * torch.sum(oc * ray.d, dim=-1)
            c_coeff = torch.sum(oc * oc, dim=-1) - R * R

            discriminant = b * b - 4 * a * c_coeff
            valid_intersect = discriminant >= 0

            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=EPSILON))
            t1 = (-b - sqrt_discriminant) / (2 * a + EPSILON)
            t2 = (-b + sqrt_discriminant) / (2 * a + EPSILON)

            # Choose intersection closest to z=0 (surface vertex)
            z1 = ray.o[..., 2] + t1 * ray.d[..., 2]
            z2 = ray.o[..., 2] + t2 * ray.d[..., 2]
            use_t1 = torch.abs(z1) < torch.abs(z2)
            t = torch.where(use_t1, t1, t2)

            new_o = ray.o + t.unsqueeze(-1) * ray.d

            # Check aperture
            r_squared = new_o[..., 0] ** 2 + new_o[..., 1] ** 2
            within_aperture = r_squared <= (self.r**2 + EPSILON)

            valid = valid_intersect & within_aperture & (ray.is_valid > 0)

        # Update ray position
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.coherent:
            if t.abs().max() > 100 and torch.get_default_dtype() == torch.float32:
                raise Exception(
                    "Using float32 may cause precision problem for OPL calculation."
                )
            new_opl = ray.opl + n * t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        valid = (x**2 + y**2) < 1 / c**2
        return valid

    def max_height(self):
        """Maximum valid height."""
        if self.tolerancing:
            c = self.c + self.c_error
        else:
            c = self.c

        max_height = torch.sqrt(1 / c**2).item() - 0.001
        return max_height

    # =========================================
    # Tolerancing
    # =========================================
    def init_tolerance(self, tolerance_params=None):
        """Initialize tolerance parameters for the surface.

        Args:
            tolerance_params (dict): Tolerance for surface parameters.
        """
        super().init_tolerance(tolerance_params)
        self.c_tole = tolerance_params.get("c_tole", 0.0001)

    def sample_tolerance(self):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)

    def zero_tolerance(self):
        """Zero tolerance."""
        super().zero_tolerance()
        self.c_error = 0.0

    def sensitivity_score(self):
        """Tolerance squared sum."""
        score_dict = super().sensitivity_score()
        score_dict.update(
            {
                "c_grad": round(self.c.grad.item(), 6),
                "c_score": round(
                    (self.c_tole**2 * self.c.grad**2).item(), 6
                ),
            }
        )
        return score_dict

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """Activate gradient computation for c and d and return optimizer parameters."""
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})
        params.append({"params": [self.c], "lr": lrs[1]})

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        roc = 1 / self.c.item() if self.c.item() != 0 else 0.0
        surf_dict = {
            "type": "Spheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(roc, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r} 1 0 0 1 ""
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str

# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

import torch

from deeplens.optics.geometric_surface.base import Surface, EPSILON


class Spiral(Surface):
    """Spiral surface.

    z(x,y) = c1 / 2 * (1 + cos(N * theta + eta * phi_norm**2)) + c2 / 2 * (1 - cos(N * theta + eta * phi_norm**2))
    """

    def __init__(self, r, d, c1, c2, mat2, N=1, eta=5, is_square=False, device="cpu"):
        """Initialize a Spiral surface.

        Args:
            r (float): Radius of the surface.
            d (float): Distance to the next surface.
            c1 (float): Parameter controlling the spiral pitch.
            c2 (float): Parameter controlling the spiral pitch.
            mat2 (str): Material of the medium after the surface.
            N (int): Parameter controlling the number of spiral arms.
            eta (float): Parameter controlling the spiral tightness.
            is_square (bool, optional): Whether the aperture is square. Defaults to False.
            device (str, optional): Device to use for torch tensors. Defaults to "cpu".
        """
        super().__init__(r, d, mat2, is_square=is_square, device=device)
        self.c1 = torch.tensor(c1, dtype=torch.float32, device=device)
        self.c2 = torch.tensor(c2, dtype=torch.float32, device=device)
        self.N = N
        self.eta = eta
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize surface from a dict."""
        return cls(
            surf_dict["r"],
            surf_dict["d"],
            surf_dict["c1"],
            surf_dict["c2"],
            surf_dict["mat2"],
            surf_dict.get("N", 1),
            surf_dict.get("eta", 5),
            surf_dict.get("is_square", False),
        )

    def _sag(self, x, y):
        """Compute surface height z(x, y) = c1 / 2 * (1 + cos(N * theta + eta * phi_norm**2)) + c2 / 2 * (1 - cos(N * theta + eta * phi_norm**2))

        Args:
            x (torch.Tensor): x coordinate.
            y (torch.Tensor): y coordinate.

        Returns:
            torch.Tensor: Surface height.

        Reference:
            [1] Spiral diopter: freeform lenses with enhanced multifocal behavior, Optica 2024.
        """
        theta = torch.atan2(y, x)  # [-pi, pi]
        phi_norm_sq = (x**2 + y**2) / self.r**2
        common_cos = torch.cos(self.N * theta + self.eta * phi_norm_sq)
        z1 = self.c1 / 2 * (1 + common_cos)
        z2 = self.c2 / 2 * (1 - common_cos)
        return z1 + z2

    def _dfdxy(self, x, y):
        """Compute surface height derivatives to x and y."""
        phi_sq = x**2 + y**2
        phi_norm_sq = phi_sq / (self.r**2 + EPSILON)
        theta = torch.atan2(y, x)

        # Argument of cosine
        u = self.N * theta + self.eta * phi_norm_sq

        # Common term: (c2-c1)/2 * sin(u)
        common_term = (self.c1 - self.c2) / 2 * (-torch.sin(u))

        # Avoid division by zero
        inv_phi_sq = 1.0 / (phi_sq + EPSILON)

        # d(u)/dx
        du_dx = -self.N * y * inv_phi_sq + 2 * self.eta * x / self.r**2
        sx = common_term * du_dx

        # d(u)/dy
        du_dy = self.N * x * inv_phi_sq + 2 * self.eta * y / self.r**2
        sy = common_term * du_dy

        return sx, sy

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-4], optim_mat=False):
        """Return parameters for optimizer."""
        params = []

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize c1
        self.c1.requires_grad_(True)
        params.append({"params": [self.c1], "lr": lrs[1]})

        # Optimize c2
        self.c2.requires_grad_(True)
        params.append({"params": [self.c2], "lr": lrs[2]})

        # We do not optimize material parameters for spiral surface.
        if optim_mat:
            raise ValueError("Material parameters are not optimized for spiral surface.")

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters."""
        s_dict = super().surf_dict()
        s_dict.update(
            {
                "c1": self.c1.item(),
                "c2": self.c2.item(),
            }
        )
        return s_dict

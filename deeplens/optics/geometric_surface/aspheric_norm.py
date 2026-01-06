# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric surface with normalized coefficients for stable optimization.

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.

Note: one problem for normalized aspheric coefficients is that when norm_r is smaller than 1, the gradient can be very large.
"""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


class AsphericNorm(Surface):
    def __init__(
        self,
        r,
        d,
        c,
        k,
        ai,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize aspheric surface.

        Args:
            r (float): radius of the surface, used for normalization
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): absolute aspheric coefficients
            mat2 (Material): material of the second medium
            device (torch.device): device to store the tensor
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
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)

        self.norm_r = r if r > 2.0 else 2.0
        self.ai = torch.tensor(ai)
        self.ai_degree = len(ai)
        # Create normalized coefficients for optimization and sag calculation
        for i, a in enumerate(ai):
            p_name = f"norm_ai{2 * (i + 1)}"
            norm_coeff = torch.tensor(a * self.norm_r ** (2 * (i + 1)))
            setattr(self, p_name, norm_coeff)

        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict["k"],
            ai=surf_dict["ai"],
            mat2=surf_dict["mat2"],
        )

    def _sag(self, x, y):
        """Compute surface height."""
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k

        # Calculate surface sag
        r2 = x**2 + y**2
        total_surface = r2 * c / (1 + torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON))

        # rho2 = r2 / self.norm_r**2
        # sag_aspheric = sum(norm_ai_{2i} * rho2**i) = sum(norm_ai_{2i} * (r2/self.norm_r**2)**i)
        # = sum (ai_{2i} * self.norm_r**(2i)) * (r2**i / self.norm_r**(2i)) = sum(ai_{2i} * r2**i)

        rho2 = r2 / (self.norm_r**2)
        for i in range(1, self.ai_degree + 1):
            norm_ai = getattr(self, f"norm_ai{2 * i}")
            total_surface += norm_ai * rho2**i

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order height derivatives to x and y."""
        # Tolerance
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k

        # Calculate surface height derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        if self.ai_degree > 0:
            # d(sag_aspheric)/dr2 = d/dr2(sum(norm_ai_{2i} * (r2/self.norm_r**2)**i))
            # = sum(norm_ai_{2i} * i * (r2/self.norm_r**2)**(i-1) * (1/self.norm_r**2))
            # = sum(norm_ai_{2i} * i * r2**(i-1) / self.norm_r**(2i))
            for i in range(1, self.ai_degree + 1):
                norm_ai = getattr(self, f"norm_ai{2 * i}")
                dsdr2 += i * norm_ai * r2 ** (i - 1) / (self.norm_r ** (2 * i))

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k

        if self.k > -1:
            valid = (x**2 + y**2) < 1 / c**2 / (1 + k)
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """Maximum valid height."""
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k

        if self.k > -1:
            max_height = torch.sqrt(1 / (k + 1) / (c**2)).item() - 0.01
        else:
            max_height = 10e3

        return max_height

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.001, optim_mat=False
    ):
        """Get optimizer parameters for different parameters.

        Args:
            lrs (list, optional): learning rates for d, c, k, and normalized ai.
            optim_mat (bool, optional): whether to optimize material. Defaults to False.
        """
        # Broadcast learning rates to all aspheric coefficients
        if self.ai_degree > 0 and len(lrs) == 4:
            lrs = lrs + [lrs[-1]] * (self.ai_degree - 1)

        params = []
        param_idx = 0

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[param_idx]})
        param_idx += 1

        # Optimize curvature
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[param_idx]})
        param_idx += 1

        # Optimize conic constant
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[param_idx]})
        param_idx += 1

        # Optimize aspheric coefficients
        if self.ai is not None:
            if self.ai_degree > 0:
                for i in range(1, self.ai_degree + 1):
                    p_name = f"norm_ai{2 * i}"
                    p = getattr(self, p_name)
                    p.requires_grad_(True)
                    params.append({"params": [p], "lr": lrs[param_idx]})
                    param_idx += 1

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    @torch.no_grad()
    def update_r(self, r):
        """Update the radius of the surface."""
        # Update radius
        r_max = self.max_height()
        r_new = max(min(r, r_max), 0.5)

        # Update normalized ai
        norm_r_old = self.norm_r
        norm_r_new = r_new if r_new > 2.0 else 2.0
        for i in range(1, self.ai_degree + 1):
            norm_ai = getattr(self, f"norm_ai{2 * i}")
            norm_ai.data = norm_ai.data * (norm_r_new / norm_r_old) ** (2 * i)

        # Update radius
        self.r = r_new
        self.norm_r = norm_r_new

    # =======================================
    # Tolerancing
    # =======================================
    @torch.no_grad()
    def init_tolerance(self, tolerance_params=None):
        """Perturb the surface with some tolerance."""
        super().init_tolerance(tolerance_params)
        self.c_tole = tolerance_params.get("c_tole", 0.0001)
        self.k_tole = tolerance_params.get("k_tole", 0.001)

    def sample_tolerance(self):
        """Sample a random manufacturing error for the surface."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)

    def zero_tolerance(self):
        """Clear perturbation."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0

    def sensitivity_score(self):
        """Tolerance squared sum."""
        score_dict = super().sensitivity_score()
        score_dict.update(
            {
                "c_grad": round(self.c.grad.item(), 6),
                "c_score": round((self.c_tole**2 * self.c.grad**2).item(), 6),
            }
        )
        score_dict.update(
            {
                "k_grad": round(self.k.grad.item(), 6),
                "k_score": round((self.k_tole**2 * self.k.grad**2).item(), 6),
            }
        )
        return score_dict

    # =======================================
    # IO
    # =======================================
    def construct_ai(self):
        for i in range(1, self.ai_degree + 1):
            p_name = f"norm_ai{2 * i}"
            norm_ai = getattr(self, p_name)
            setattr(self, f"ai{2 * i}", norm_ai / (self.norm_r ** (2 * i)))

    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "Aspheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(1 / self.c.item(), 4),
            "d": round(self.d.item(), 4),
            "k": round(self.k.item(), 4),
            "ai": [],
            "mat2": self.mat2.get_name(),
        }
        if self.ai_degree > 0:
            for i in range(1, self.ai_degree + 1):
                p_name = f"norm_ai{2 * i}"
                norm_ai = getattr(self, p_name)
                abs_ai = norm_ai.item() / (self.norm_r ** (2 * i))
                surf_dict[f"(ai{2 * i})"] = float(format(abs_ai, ".6e"))
                surf_dict["ai"].append(float(format(abs_ai, ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )

        # Get absolute ai values for Zemax file
        abs_ai = []
        if self.ai_degree > 0:
            for i in range(1, self.ai_degree + 1):
                norm_ai = getattr(self, f"norm_ai{2 * i}")
                abs_ai.append(norm_ai.item() / (self.norm_r ** (2 * i)))

        # Pad with zeros if necessary for Zemax PARM format
        while len(abs_ai) < 6:
            abs_ai.append(0.0)

        common_params = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {d_next.item()}"""

        if self.mat2.get_name() != "air":
            common_params += f"""
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}"""

        common_params += f"""
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}"""

        return common_params

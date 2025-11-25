# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric surface.

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


class Aspheric(Surface):
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
            r (float): radius of the surface
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): aspheric coefficients
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
        if ai is not None:
            self.ai = torch.tensor(ai)
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
            elif self.ai_degree == 5:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
                self.ai10 = torch.tensor(ai[4])
            elif self.ai_degree == 6:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
                self.ai10 = torch.tensor(ai[4])
                self.ai12 = torch.tensor(ai[5])
            else:
                for i, a in enumerate(ai):
                    exec(f"self.ai{2 * i + 2} = torch.tensor({a})")
        else:
            self.ai = None
            self.ai_degree = 0

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

        # Calculate surface height
        r2 = x**2 + y**2
        total_surface = r2 * c / (1 + torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON))

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                )
            elif self.ai_degree == 5:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                    + self.ai10 * r2**5
                )
            elif self.ai_degree == 6:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                    + self.ai10 * r2**5
                    + self.ai12 * r2**6
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"total_surface += self.ai{2 * i} * r2 ** {i}")

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

        # Compute surface height derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                )
            elif self.ai_degree == 5:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                )
            elif self.ai_degree == 6:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                    + 6 * self.ai12 * r2**5
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"dsdr2 += {i} * self.ai{2 * i} * r2 ** {i - 1}")

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k

        if k > -1:
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

        if k > -1:
            max_height = torch.sqrt(1 / (k + 1) / (c**2)).item() - 0.001
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
            lrs (list, optional): learning rates for d, c, k, ai2, (ai4, ai6, ai8, ai10, ai12).
            optim_mat (bool, optional): whether to optimize material. Defaults to False.
        """
        # Broadcast learning rates to all aspheric coefficients
        if len(lrs) == 4:
            lrs = lrs + [
                lrs[-1] * decay ** (ai_degree + 1)
                for ai_degree in range(self.ai_degree - 1)
            ]

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
                    p_name = f"ai{2 * i}"
                    p = getattr(self, p_name)
                    p.requires_grad_(True)
                    params.append({"params": [p], "lr": lrs[param_idx]})
                    param_idx += 1

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # Tolerancing
    # =======================================

    @torch.no_grad()
    def init_tolerance(self, tolerance_params=None):
        """Perturb the surface with some tolerance.

        Args:
            tolerance_params (dict): Tolerance for surface parameters.

        References:
            [1] https://www.edmundoptics.com/capabilities/precision-optics/capabilities/aspheric-lenses/
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/all-about-aspheric-lenses/?srsltid=AfmBOoon8AUXVALojol2s5K20gQk7W1qUisc6cE4WzZp3ATFY5T1pK8q
        """
        super().init_tolerance(tolerance_params)
        self.c_tole = tolerance_params.get("c_tole", 0.001)
        self.k_tole = tolerance_params.get("k_tole", 0.001)

    def sample_tolerance(self):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)

    def zero_tolerance(self):
        """Zero tolerance."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0

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

        score_dict.update(
            {
                "k_grad": round(self.k.grad.item(), 6),
                "k_score": round(
                    (self.k_tole**2 * self.k.grad**2).item(), 6
                ),
            }
        )
        return score_dict

    # =======================================
    # IO
    # =======================================
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
        for i in range(1, self.ai_degree + 1):
            exec(
                f"surf_dict['(ai{2 * i})'] = float(format(self.ai{2 * i}.item(), '.6e'))"
            )
            surf_dict["ai"].append(float(format(eval(f"self.ai{2 * i}.item()"), ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        return zmx_str

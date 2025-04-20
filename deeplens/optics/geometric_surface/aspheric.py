"""Aspheric surface.

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import numpy as np
import torch

from .base import EPSILON, Surface


class Aspheric(Surface):
    def __init__(self, r, d, c=0.0, k=0.0, ai=None, mat2=None, device="cpu"):
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
        Surface.__init__(self, r, d, mat2, is_square=False, device=device)
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

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        if "ai" in surf_dict:
            ai = surf_dict["ai"]
        else:
            ai = torch.rand(6) * 1e-30

        return cls(
            surf_dict["r"], surf_dict["d"], c, surf_dict["k"], ai, surf_dict["mat2"]
        )

    def _sag(self, x, y):
        """Compute surface height."""
        r2 = x**2 + y**2
        total_surface = (
            r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON))
        )

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
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON)
        dsdr2 = (
            (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf) ** 2
        )

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

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of surface height with respect to x and y."""
        r2 = x**2 + y**2
        c = self.c
        k = self.k
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)

        # Compute dsdr2
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / (2 * sf)) * c / (1 + sf) ** 2

        # Compute derivative of dsdr2 with respect to r2 (ddsdr2_dr2)
        ddsdr2_dr2 = (
            ((1 + k) * c**2 / (2 * sf)) + ((1 + k) ** 2 * r2 * c**4) / (4 * sf**3)
        ) * c / (1 + sf) ** 2 - 2 * dsdr2 * (
            1 + sf + (1 + k) * r2 * c**2 / (2 * sf)
        ) / (1 + sf)

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                )
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
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
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
                    + 20 * self.ai10 * r2**3
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
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
                    + 20 * self.ai10 * r2**3
                    + 30 * self.ai12 * r2**4
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    ai_coeff = getattr(self, f"ai{2 * i}")
                    dsdr2 += i * ai_coeff * r2 ** (i - 1)
                    if i > 1:
                        ddsdr2_dr2 += i * (i - 1) * ai_coeff * r2 ** (i - 2)
        else:
            ddsdr2_dr2 = ddsdr2_dr2

        # Compute second-order derivatives
        d2f_dx2 = 2 * dsdr2 + 4 * x**2 * ddsdr2_dr2
        d2f_dxdy = 4 * x * y * ddsdr2_dr2
        d2f_dy2 = 2 * dsdr2 + 4 * y**2 * ddsdr2_dr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        if self.k > -1:
            valid = (x**2 + y**2) < 1 / self.c**2 / (1 + self.k)
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """Maximum valid height."""
        if self.k > -1:
            max_height = torch.sqrt(1 / (self.k + 1) / (self.c**2)).item() - 0.01
        else:
            max_height = 100

        return max_height

    
    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(
        self, lr=[1e-4, 1e-4, 1e-1, 1e-2], decay=0.01, optim_mat=False
    ):
        """Get optimizer parameters for different parameters.

        Args:
            lr (list, optional): learning rates for c, d, k, ai. Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr * 1e3, lr]

        params = []
        if lr[0] > 0 and self.c != 0:
            self.c.requires_grad_(True)
            params.append({"params": [self.c], "lr": lr[0]})

        if lr[1] > 0:
            self.d.requires_grad_(True)
            params.append({"params": [self.d], "lr": lr[1]})

        if lr[2] > 0 and self.k != 0:
            self.k.requires_grad_(True)
            params.append({"params": [self.k], "lr": lr[2]})

        if lr[3] > 0:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
            elif self.ai_degree == 5:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
                params.append({"params": [self.ai10], "lr": lr[3] * decay**4})
            elif self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
                params.append({"params": [self.ai10], "lr": lr[3] * decay**4})
                params.append({"params": [self.ai12], "lr": lr[3] * decay**5})
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"self.ai{2 * i}.requires_grad_(True)")
                    exec(
                        f"params.append({{'params': [self.ai{2 * i}], 'lr': lr[3] * decay**{i - 1}}})"
                    )

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params
    

    # =======================================
    # Manufacturing
    # =======================================
    @torch.no_grad()
    def perturb(self, tolerance):
        """Randomly perturb surface parameters to simulate manufacturing errors.

        Args:
            tolerance (dict): Tolerance for surface parameters.
        """
        self.r_offset = float(self.r * np.random.randn() * tolerance.get("r", 0.001))
        self.c_offset = float(self.c * np.random.randn() * tolerance.get("c", 0.001))
        self.d_offset = float(np.random.randn() * tolerance.get("d", 0.001))
        self.k_offset = float(np.random.randn() * tolerance.get("k", 0.001))
        for i in range(1, self.ai_degree + 1):
            exec(
                f"self.ai{2 * i}_offset = float(np.random.randn() * tolerance.get('ai{2 * i}', 0.001))"
            )

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

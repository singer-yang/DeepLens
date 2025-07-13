"""Aspheric surface with normalized coefficients for stable optimization.

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.

Note: one problem for normalized aspheric coefficients is that when norm_r is smaller than 1, the gradient can be very large.
"""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


class AsphericNorm(Surface):
    def __init__(self, r, d, c=0.0, k=0.0, ai=None, mat2=None, device="cpu"):
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
        Surface.__init__(self, r, d, mat2, is_square=False, device=device)
        self.norm_r = r if r > 2.0 else 2.0
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        if ai is not None:
            self.ai = torch.tensor(ai) # Store absolute ai
            self.ai_degree = len(ai)
            # Create normalized coefficients for optimization and sag calculation
            for i, a in enumerate(ai):
                p_name = f"norm_ai{2 * (i + 1)}"
                # abs_ai = norm_ai / norm_r^(2(i+1))
                norm_coeff = torch.tensor(a * self.norm_r ** (2 * (i + 1)))
                setattr(self, p_name, norm_coeff)
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
        
        # rho2 = r2 / self.norm_r**2
        # sag_aspheric = sum(norm_ai_{2i} * rho2**i) = sum(norm_ai_{2i} * (r2/self.norm_r**2)**i)
        # = sum (ai_{2i} * self.norm_r**(2i)) * (r2**i / self.norm_r**(2i)) = sum(ai_{2i} * r2**i)

        rho2 = r2 / (self.norm_r**2)
        for i in range(1, self.ai_degree + 1):
            norm_ai = getattr(self, f"norm_ai{2 * i}")
            total_surface += norm_ai * rho2 ** i

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order height derivatives to x and y."""
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON)
        dsdr2 = (
            (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf) ** 2
        )

        if self.ai_degree > 0:
            # d(sag_aspheric)/dr2 = d/dr2(sum(norm_ai_{2i} * (r2/self.norm_r**2)**i))
            # = sum(norm_ai_{2i} * i * (r2/self.norm_r**2)**(i-1) * (1/self.norm_r**2))
            # = sum(norm_ai_{2i} * i * r2**(i-1) / self.norm_r**(2i))
            for i in range(1, self.ai_degree + 1):
                norm_ai = getattr(self, f"norm_ai{2 * i}")
                dsdr2 += i * norm_ai * r2 ** (i - 1) / (self.norm_r ** (2 * i))

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of surface height with respect to x and y."""
        r2 = x**2 + y**2
        c = self.c
        k = self.k
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)

        # 1. Compute the full first derivative dsdr2
        # Conic part
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / (2 * sf)) * c / (1 + sf) ** 2
        
        # Aspheric part
        if self.ai_degree > 0:
            # d(sag_aspheric)/dr2 = d/dr2(sum(norm_ai_{2i} * (r2/self.norm_r**2)**i))
            # = sum(norm_ai_{2i} * i * r2**(i-1) / self.norm_r**(2i))
            for i in range(1, self.ai_degree + 1):
                norm_ai = getattr(self, f"norm_ai{2 * i}")
                dsdr2 += i * norm_ai * r2 ** (i - 1) / (self.norm_r ** (2 * i))

        # 2. Compute the full second derivative ddsdr2_dr2
        # This is the derivative of dsdr2 with respect to r2.
        # Conic part's contribution to the second derivative.
        # This formula correctly depends on the *full* dsdr2 value computed above.
        ddsdr2_dr2 = (
            (((1 + k) * c**2) / (2 * sf) + ((1 + k)**2 * r2 * c**4) / (4 * sf**3)) * c / (1 + sf)**2
            - (2 * dsdr2 * ((1 + k) * c**2 / (2 * sf))) / (1 + sf)
        )
        
        # Aspheric part's contribution to the second derivative
        if self.ai_degree > 1:
            # d/dr2 of aspheric part of dsdr2: sum(i * (i-1) * norm_ai_{2i} * r2**(i-2) / self.norm_r**(2i))
            for i in range(2, self.ai_degree + 1):
                norm_ai = getattr(self, f"norm_ai{2 * i}")
                ddsdr2_dr2 += i * (i - 1) * norm_ai * r2 ** (i - 2) / (self.norm_r ** (2 * i))

        # 3. Compute final second-order derivatives
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
    def get_optim_param_count(self, optim_mat=False):
        """Get number of optimizable parameters."""
        count = 0
        if self.c != 0:
            count += 1
        count += 1  # for d
        if self.k != 0:
            count += 1
        
        if self.ai is not None:
            count += self.ai_degree

        if optim_mat and self.mat2.get_name() != "air":
            count += self.mat2.get_optim_param_count()
        return count

    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.001, optim_mat=False):
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
                    p_name = f"norm_ai{2*i}"
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
        r_new = r
        norm_r_old = self.norm_r
        norm_r_new = r_new if r_new > 2.0 else 2.0
        
        # Update normalized ai
        for i in range(1, self.ai_degree + 1):
            norm_ai = getattr(self, f"norm_ai{2 * i}")
            norm_ai.data = norm_ai.data * (norm_r_new / norm_r_old) ** (2 * i)

        # Update radius
        self.r = r_new
        self.norm_r = norm_r_new

    # =======================================
    # Perturbation
    # =======================================
    @torch.no_grad()
    def perturb(self, tolerance):
        """Perturb the surface with some tolerance."""
        self.r_offset = float(self.r * np.random.randn() * tolerance.get("r", 0.001))
        self.c_offset = float(self.c * np.random.randn() * tolerance.get("c", 0.001))
        self.d_offset = float(np.random.randn() * tolerance.get("d", 0.001))
        self.k_offset = float(np.random.randn() * tolerance.get("k", 0.001))
        for i in range(1, self.ai_degree + 1):
            p_name = f"norm_ai{2 * i}"
            offset_name = f"{p_name}_offset"
            offset_val = float(np.random.randn() * tolerance.get(p_name, 0.001))
            setattr(self, offset_name, offset_val)

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
        if self.ai_degree > 0:
            for i in range(1, self.ai_degree + 1):
                p_name = f"norm_ai{2 * i}"
                norm_ai = getattr(self, p_name)
                # abs_ai = norm_ai / norm_r^(2i)
                abs_ai = norm_ai.item() / (self.norm_r ** (2 * i))
                surf_dict[f'(ai{2 * i})'] = float(format(abs_ai, '.6e'))
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

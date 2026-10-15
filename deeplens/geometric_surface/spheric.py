# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Spheric surface."""

import torch

from .base_surface import EPSILON, Surface


class Spheric(Surface):
    """Spherical refractive surface parameterized by curvature.

    A sphere of radius $R = 1/c$ whose vertex sits at the optical axis. The
    sag (surface height along $z$) is:

    $$
    z(x, y) = \\frac{c \\rho^2}{1 + \\sqrt{1 - c^2 \\rho^2}}, \\quad
    \\rho^2 = x^2 + y^2
    $$

    Attributes:
        c (torch.Tensor): Surface curvature $1/R$ [1/mm], scalar tensor.
            Gradients are enabled by `get_optimizer_params` for optimization.
    """

    def __init__(
        self,
        c,
        r,
        d_next,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize a spherical surface.

        Args:
            c (float): Surface curvature $1/R$ [1/mm]. Use 0 for a flat
                surface (treated as a plane).
            r (float): Aperture radius [mm].
            d (float): Axial vertex position [mm].
            mat2 (str or Material): Material on the transmission side.
            pos_xy (list[float], optional): Lateral offset `[x, y]` [mm].
                Defaults to `[0.0, 0.0]`.
            vec_local (list[float], optional): Local surface normal direction.
                Defaults to `[0.0, 0.0, 1.0]`.
            is_square (bool, optional): Use a square aperture instead of a
                circular one. Defaults to False.
            device (str, optional): Compute device. Defaults to `"cpu"`.
        """
        super(Spheric, self).__init__(
            r=r,
            d_next=d_next,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.c = torch.as_tensor(c, dtype=self.d_next.dtype, device=device)
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a `Spheric` surface from a parameter dictionary.

        Accepts either a radius of curvature `roc` [mm] (converted to curvature
        `c`, with `roc == 0` mapped to `c = 0`) or a curvature `c` [1/mm]
        directly. The aperture radius `r` [mm], vertex position `d` [mm], and
        transmission material `mat2` are read from the dictionary.

        Args:
            surf_dict (dict): Surface parameters. Must contain `r`, `d`, `mat2`
                and either `roc` or `c`.

        Returns:
            surface (Spheric): The constructed spherical surface.
        """
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
            d_next=surf_dict["d_next"],
            mat2=surf_dict["mat2"],
            pos_xy=surf_dict.get("pos_xy", [0.0, 0.0]),
            vec_local=surf_dict.get("vec_local", [0.0, 0.0, 1.0]),
            is_square=surf_dict.get("is_square", False),
            device=surf_dict.get("device", "cpu"),
        )

    def paraxial_power(self, n1, n2, wvln=None):
        """Return the paraxial optical power of this surface [1/mm].

        Only the vertex curvature contributes; higher-order geometry does
        not affect first-order properties.

        Args:
            n1 (torch.Tensor): Refractive index of the incident medium.
            n2 (torch.Tensor): Refractive index of the transmission medium.
            wvln (float, optional): Wavelength [µm]. Unused by refractive
                surfaces, which see it only through `n1`/`n2`.

        Returns:
            power (torch.Tensor): Surface power `(n2 - n1) * c` [1/mm], scalar.
        """
        return (n2 - n1) * self.c

    def _sag(self, x, y):
        """Compute surface sag $z = c\\rho^2 / (1 + \\sqrt{1 - c^2\\rho^2})$.

        Here $\\rho^2 = x^2 + y^2$. The radicand is clamped to `EPSILON` to keep
        the square root finite beyond the valid radius.

        Args:
            x (torch.Tensor): Local x coordinate [mm].
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            sag (torch.Tensor): Surface sag along z [mm], same shape as `x`.
        """
        c = self.c

        # Compute surface sag
        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON)))
        return sag

    def _dfdxy(self, x, y):
        """Compute first-order sag derivatives $\\partial z/\\partial x$ and $\\partial z/\\partial y$.

        Args:
            x (torch.Tensor): Local x coordinate [mm].
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            dfdx (torch.Tensor): Sag derivative w.r.t. x [dimensionless], same shape as `x`.
            dfdy (torch.Tensor): Sag derivative w.r.t. y [dimensionless], same shape as `x`.
        """
        c = self.c

        # Compute surface sag derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON))
        dfdr2 = c / (2 * sf)

        dfdx = dfdr2 * 2 * x
        dfdy = dfdr2 * 2 * y

        return dfdx, dfdy

    def _d2fdxy(self, x, y):
        """Compute second-order sag derivatives via the chain rule.

        Args:
            x (torch.Tensor): Local x coordinate [mm].
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            d2f_dx2 (torch.Tensor): $\\partial^2 z/\\partial x^2$ [1/mm], same shape as `x`.
            d2f_dxdy (torch.Tensor): $\\partial^2 z/\\partial x\\partial y$ [1/mm], same shape as `x`.
            d2f_dy2 (torch.Tensor): $\\partial^2 z/\\partial y^2$ [1/mm], same shape as `x`.
        """
        c = self.c

        # Compute surface sag derivatives
        r2 = x**2 + y**2
        sf = torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON))

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
        """Solve the ray-surface intersection analytically in local coordinates.

        Substitutes the ray $p(t) = o + t\\,d$ into the sphere
        $x^2 + y^2 + (z - R)^2 = R^2$ (with $R = 1/c$) and solves the resulting
        quadratic for $t$, picking the root whose intersection lies closest to
        the surface vertex at $z = 0$. A flat surface ($|c| < $ `EPSILON`) is
        handled as a plane. Rays falling outside the aperture or with no real
        root are flagged invalid. Updates `ray.o`, `ray.is_valid`, and, for
        coherent rays, `ray.opl` (adding $n\\,t$).

        Args:
            ray (Ray): Input ray, modified in place.
            n (float, optional): Refractive index of the incident medium, used
                for the optical path length update. Defaults to 1.0.

        Returns:
            ray (Ray): The same ray with updated position, validity, and opl.

        Raises:
            Exception: If a coherent ray travels more than 100 mm under
                float32, where OPL accumulation loses precision.
        """
        c = self.c
        original_o = ray.o

        # Use the vertex-anchored sphere equation. The surface passes through
        # the local origin, so
        #
        #   x² + y² + z² - 2 z R = 0,  R = 1 / c.
        #
        # Multiplying by c before substituting o + t*d gives a quadratic with
        # no R² subtraction. Flat surfaces are selected with tensor operations
        # so tracing does not extract curvature to the host at every surface.
        is_flat = torch.abs(c) < EPSILON
        c_solve = torch.where(is_flat, torch.ones_like(c), c)
        od = torch.sum(original_o * ray.d, dim=-1)
        dd = torch.sum(ray.d * ray.d, dim=-1)
        oo = torch.sum(original_o * original_o, dim=-1)

        a = c_solve * dd
        b = 2.0 * (c_solve * od - ray.d[..., 2])
        c_coeff = c_solve * oo - 2.0 * original_o[..., 2]

        discriminant = b * b - 4 * a * c_coeff
        sphere_valid = discriminant >= 0
        sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=EPSILON))

        # Stable root pair. q avoids subtracting nearly equal values; q/a and
        # c_coeff/q are the two roots by Vieta's formula.
        q = torch.where(
            b >= 0,
            -(b + sqrt_discriminant) / 2.0,
            (sqrt_discriminant - b) / 2.0,
        )
        q_safe = torch.where(
            q.abs() < EPSILON,
            torch.where(
                q < 0,
                -torch.full_like(q, EPSILON),
                torch.full_like(q, EPSILON),
            ),
            q,
        )
        t1 = q / a
        t2 = c_coeff / q_safe

        z1 = original_o[..., 2] + t1 * ray.d[..., 2]
        z2 = original_o[..., 2] + t2 * ray.d[..., 2]
        sphere_t = torch.where(torch.abs(z1) < torch.abs(z2), t1, t2)

        dz_safe = torch.where(
            ray.d[..., 2].abs() < EPSILON,
            torch.where(
                ray.d[..., 2] < 0,
                -torch.full_like(ray.d[..., 2], EPSILON),
                torch.full_like(ray.d[..., 2], EPSILON),
            ),
            ray.d[..., 2],
        )
        plane_t = -original_o[..., 2] / dz_safe
        t = torch.where(is_flat, plane_t, sphere_t)
        plane_valid = (ray.d[..., 2].abs() >= EPSILON) & torch.isfinite(plane_t)
        valid_intersect = torch.where(is_flat, plane_valid, sphere_valid)

        new_o = original_o + t.unsqueeze(-1) * ray.d
        within_aperture = self.is_within_boundary(new_o[..., 0], new_o[..., 1])
        valid = (
            valid_intersect & torch.isfinite(t) & within_aperture & (ray.is_valid > 0)
        )

        # Update ray position
        ray.o = torch.where(valid.unsqueeze(-1), new_o, original_o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            total_t = t
            if ray.o.dtype == torch.float32 and total_t.abs().max() > 100:
                raise ValueError(
                    "Coherent ray tracing over long paths requires float64 rays."
                )
            new_opl = ray.opl + n * total_t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def is_within_data_range(self, x, y):
        """Check whether points lie within the sag-defined region.

        Points are valid only where $x^2 + y^2 < 1/c^2$, i.e. inside the radius
        where the sphere's sag is real-valued.

        Args:
            x (torch.Tensor): Local x coordinate [mm].
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            valid (torch.Tensor): Boolean mask, same shape as `x`.
        """
        c = self.c
        valid = (x**2 + y**2) < 1 / c**2
        return valid

    def max_height(self):
        """Return the maximum valid radial height of the surface.

        Equal to $|R| = 1/|c|$ minus a small 0.001 mm margin to stay inside the
        region where the sag is well-defined.

        Returns:
            max_height (float): Maximum radial height [mm].
        """
        c = self.c
        max_height = torch.sqrt(1 / c**2).item() - 0.001
        return max_height

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """Enable gradients on `c` and `d` and build optimizer parameter groups.

        Args:
            lrs (list[float], optional): Learning rates `[lr_d, lr_c]` for the
                vertex position and curvature. Defaults to `[1e-4, 1e-4]`.
            optim_mat (bool, optional): Also optimize the transmission material
                parameters (skipped when the material is air). Defaults to False.

        Returns:
            params (list[dict]): Optimizer parameter groups, each with `params`
                and `lr` keys.
        """
        self.c.requires_grad_(True)
        self.d_next.requires_grad_(True)

        params = []
        params.append({"params": [self.d_next], "lr": lrs[0]})
        params.append({"params": [self.c], "lr": lrs[1]})

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Serialize the surface to a parameter dictionary.

        Returns:
            surf_dict (dict): Surface parameters with keys `type`, `r`, `(c)`,
                `roc`, `(d)`, `mat2`, plus informational `(mat2_n)`/`(mat2_V)`.
                Lengths are in [mm], curvature in
                [1/mm], rounded to 4 decimals.
        """
        c_value = self.c.item()
        roc = 1 / c_value if c_value != 0 else 0.0
        surf_dict = {
            "type": "Spheric",
            "r": self.r,
            "(c)": c_value,
            "roc": roc,
            "d_next": self.d_next.item(),
            "pos_xy": [self.pos_x.item(), self.pos_y.item()],
            "vec_local": self.vec_local.tolist(),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Format the surface as a Zemax STANDARD surface block.

        Args:
            surf_idx (int): Surface index in the Zemax file.
            d_next (torch.Tensor): Axial distance to the next surface [mm],
                scalar tensor.

        Returns:
            zmx_str (str): Multi-line Zemax surface description.
        """
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
    GLAS {self.mat2.zmx_glass()}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str

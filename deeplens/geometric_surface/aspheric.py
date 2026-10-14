# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric surface.

The `ai` coefficient list starts from the 4th-order term (a4) by default.
Legacy JSON files that include a 2nd-order term (a2) are loaded via the
`use_ai2` flag in `init_from_dict`. When present, `a2` is stored
separately and included in the sag computation but is **not** optimised
(it competes with the base curvature `c`).

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import torch

from .base_surface import EPSILON, Surface


class Aspheric(Surface):
    """Even-order aspheric surface.

    The sag function is:

    $$
    z(\\rho) = \\frac{c\\,\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}}
    + \\sum_{i=2}^{n} a_{2i}\\,\\rho^{2i},
    \\quad \\rho^2 = x^2 + y^2
    $$

    The polynomial starts at the 4th-order term (a4) because the 2nd-order
    term competes with the base curvature `c`.

    All coefficients `c`, `k`, and `ai` are differentiable torch
    tensors so they can be optimised with gradient descent.

    Attributes:
        c (torch.Tensor): Base curvature [1/mm].
        k (torch.Tensor): Conic constant.
        ai2 (torch.Tensor or None): 2nd-order aspheric coefficient (legacy).
        ai (torch.Tensor): Even-order aspheric coefficients
            `[a4, a6, a8, ...]`.
    """

    def __init__(
        self,
        r,
        d_next,
        c,
        k,
        ai,
        mat2,
        ai2=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize an aspheric surface.

        Args:
            r (float): Aperture radius [mm].
            d (float): Axial vertex position [mm].
            c (float): Base curvature `1/R` [1/mm].
            k (float): Conic constant (`0` = sphere, `-1` = paraboloid).
            ai (list[float] or None): Even-order aspheric coefficients
                starting from the 4th-order term: `[a4, a6, a8, ...]`.
                Pass `None` or an empty list for a pure conic.
            mat2 (str or Material): Material on the transmission side.
            ai2 (float or None, optional): 2nd-order aspheric coefficient
                from legacy data. Included in sag but not optimised.
                Defaults to None.
            pos_xy (list[float], optional): Lateral offset `[x, y]` [mm].
                Defaults to `[0.0, 0.0]`.
            vec_local (list[float], optional): Local normal direction.
                Defaults to `[0.0, 0.0, 1.0]`.
            is_square (bool, optional): Square aperture flag.
                Defaults to False.
            device (str, optional): Compute device. Defaults to `"cpu"`.
        """
        Surface.__init__(
            self,
            r=r,
            d_next=d_next,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        tensor_kwargs = {"dtype": self.d_next.dtype, "device": device}
        self.c = torch.as_tensor(c, **tensor_kwargs)
        self.k = torch.as_tensor(k, **tensor_kwargs)

        # 2nd-order coefficient (legacy, not optimised)
        if ai2 is not None:
            self.ai2 = torch.as_tensor(ai2, **tensor_kwargs)
        else:
            self.ai2 = None

        if ai is not None and len(ai) > 0:
            self.ai = torch.as_tensor(ai, **tensor_kwargs)
            self.ai_degree = len(ai)
            # ai[0] -> ai4, ai[1] -> ai6, ai[2] -> ai8, ...
            for i, a in enumerate(ai):
                setattr(self, f"ai{2 * (i + 2)}", torch.as_tensor(a, **tensor_kwargs))
        else:
            self.ai = None
            self.ai_degree = 0

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Create an aspheric surface from a serialized dict.

        The base curvature is read from `roc` (radius of curvature [mm],
        converted to `c = 1/roc`) when present, otherwise from `c` [1/mm].
        For legacy data where `use_ai2` is True (or absent), the first
        element of `ai` is interpreted as the 2nd-order coefficient `a2`
        and the rest as `[a4, a6, a8, ...]`.

        Args:
            surf_dict (dict): Serialized surface with keys `r`, `d`, `k`,
                `mat2`, either `roc` or `c`, and optionally `ai` and
                `use_ai2`.

        Returns:
            surface (Aspheric): The reconstructed aspheric surface.
        """
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        ai = surf_dict.get("ai", [])
        ai2_val = None

        # Backward compatibility: old format includes a2 as first element.
        # New files written by this code set use_ai2 explicitly.
        if surf_dict.get("use_ai2", True) and len(ai) > 0:
            if "use_ai2" not in surf_dict:
                print(
                    f"Surface dict lacks 'use_ai2'; assuming ai[0]={ai[0]:.4g} is the "
                    "2nd-order coefficient (legacy format)."
                )
            ai2_val = ai[0]  # Extract the a2 coefficient
            ai = ai[1:]  # Remaining: [a4, a6, a8, ...]

        return cls(
            r=surf_dict["r"],
            d_next=surf_dict["d_next"],
            c=c,
            k=surf_dict["k"],
            ai=ai,
            ai2=ai2_val,
            mat2=surf_dict["mat2"],
            pos_xy=surf_dict.get("pos_xy", [0.0, 0.0]),
            vec_local=surf_dict.get("vec_local", [0.0, 0.0, 1.0]),
            is_square=surf_dict.get("is_square", False),
            device=surf_dict.get("device", "cpu"),
        )

    def _get_curvature_params(self):
        """Return base curvature `c` [1/mm] and conic constant `k`.

        Returns:
            c (torch.Tensor): Base curvature [1/mm].
            k (torch.Tensor): Conic constant.
        """
        return self.c, self.k

    def newtons_method(self, ray):
        """Find an intersection, rejecting rays entering behind the aperture rim."""
        t, valid = super().newtons_method(ray)
        return t, valid & ~self._enters_through_edge(ray)

    @torch.no_grad()
    def _enters_through_edge(self, ray):
        """Identify rays entering the aperture or conic domain behind the face.

        The entry point is pulled radially inside the boundary before evaluating
        sag, so a grazing ray is not sampled past its intersection. Rays starting
        inside the valid region retain the usual Newton validity checks.

        Args:
            ray (Ray): Input ray bundle in local surface coordinates.

        Returns:
            mask (torch.Tensor): Boolean edge-entry mask, shape [...].
        """
        o, d = ray.o, ray.d
        c, k = self._get_curvature_params()
        aperture_sq = (self.w**2 + self.h**2) / 4 if self.is_square else self.r**2
        conic_power = (1 + k) * c**2
        domain_sq = torch.where(
            conic_power > 0,
            1 / torch.where(conic_power > 0, conic_power, 1.0),
            torch.full_like(conic_power, float("inf")),
        )
        radius_sq = domain_sq.clamp(max=aperture_sq).to(o)
        a = d[..., :2].square().sum(-1)
        b = (o[..., :2] * d[..., :2]).sum(-1)
        outside = o[..., :2].square().sum(-1) - radius_sq
        disc = b.square() - a * outside
        axial = a == 0
        denominator = torch.where(axial, 1.0, a)
        root = disc.clamp(min=0).sqrt()
        t_in = torch.where(axial, -float("inf"), (-b - root) / denominator)
        t_out = torch.where(axial, float("inf"), (-b + root) / denominator)
        reaches = torch.where(axial, outside <= 0, disc > 0)
        starts_inside = outside <= 0

        if self.is_square:
            # Square apertures use half-side lengths, not a circular substitute.
            half_size = o.new_tensor([self.w / 2, self.h / 2])
            parallel = d[..., :2] == 0
            safe_d = torch.where(parallel, 1.0, d[..., :2])
            t1 = (-half_size - o[..., :2]) / safe_d
            t2 = (half_size - o[..., :2]) / safe_d
            near = torch.where(parallel, -float("inf"), torch.minimum(t1, t2))
            far = torch.where(parallel, float("inf"), torch.maximum(t1, t2))
            t_in = torch.maximum(t_in, near.amax(-1))
            t_out = torch.minimum(t_out, far.amin(-1))
            inside_square = o[..., :2].abs() <= half_size
            reaches = reaches & (~parallel | inside_square).all(-1)
            starts_inside = starts_inside & inside_square.all(-1)

        enters = reaches & ~starts_inside & (t_in > 0) & (t_out > t_in)
        entry = o + d * torch.where(enters, t_in, 0.0).unsqueeze(-1)
        xy = torch.where(enters.unsqueeze(-1), entry[..., :2], 0.0) * (1 - 1e-6)
        behind = (self._sag(xy[..., 0], xy[..., 1]) - entry[..., 2]) * d[..., 2] < 0
        return enters & behind

    def paraxial_power(self, n1, n2):
        """Return the paraxial optical power of this surface [1/mm].

        Only the vertex curvature contributes, so the conic constant and the
        4th- and higher-order coefficients do not affect first-order
        properties. The legacy $a_2\\rho^2$ term is the exception: near the
        vertex the sag is $(c/2 + a_2)\\rho^2$, so it shifts the vertex
        curvature to $c + 2 a_2$ and must be included.

        Args:
            n1 (torch.Tensor): Refractive index of the incident medium.
            n2 (torch.Tensor): Refractive index of the transmission medium.

        Returns:
            power (torch.Tensor): Surface power `(n2 - n1) * (c + 2 * a2)`
                [1/mm], scalar.
        """
        c = self.c if self.ai2 is None else self.c + 2.0 * self.ai2
        return (n2 - n1) * c

    def _sag(self, x, y):
        """Compute surface sag (axial height) $z = \\mathrm{sag}(x, y)$.

        The even-order aspheric sag is

        $$
        z = \\frac{c\\,\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}}
        + a_2\\rho^2 + \\sum_{i\\ge 2} a_{2i}\\,\\rho^{2i},
        \\quad \\rho^2 = x^2 + y^2,
        $$

        where the $a_2\\rho^2$ term is only added when the legacy `ai2`
        coefficient is present. All lengths are in [mm].

        Args:
            x (torch.Tensor): x coordinate(s) [mm], any shape.
            y (torch.Tensor): y coordinate(s) [mm], broadcastable with `x`.

        Returns:
            z (torch.Tensor): Surface sag [mm], same shape as the broadcast
                of `x` and `y`.
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        total_surface = r2 * c / (1 + torch.sqrt(sf_arg))

        # Legacy a2 term: a2 * r²
        if self.ai2 is not None:
            total_surface = total_surface + self.ai2 * r2

        # Horner form: r⁴ * (a4 + r² * (a6 + r² * (...))).
        if self.ai_degree > 0:
            polynomial = getattr(self, f"ai{2 * (self.ai_degree + 1)}")
            for i in range(self.ai_degree - 2, -1, -1):
                polynomial = getattr(self, f"ai{2 * (i + 2)}") + r2 * polynomial
            total_surface = total_surface + r2 * r2 * polynomial

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order sag derivatives $\\partial z/\\partial x$ and $\\partial z/\\partial y$.

        Differentiates the sag via $\\partial z/\\partial x = (\\partial z/\\partial \\rho^2)\\,2x$.
        For the polynomial $\\sum_{i\\ge 2} a_{2i}\\rho^{2i}$, the derivative
        w.r.t. $\\rho^2$ is $\\sum_{i\\ge 2} i\\,a_{2i}\\,\\rho^{2(i-1)}$, i.e.
        $2a_4\\rho^2 + 3a_6\\rho^4 + \\dots$

        Args:
            x (torch.Tensor): x coordinate(s) [mm], any shape.
            y (torch.Tensor): y coordinate(s) [mm], broadcastable with `x`.

        Returns:
            dfdx (torch.Tensor): $\\partial z/\\partial x$ [dimensionless], same shape as input.
            dfdy (torch.Tensor): $\\partial z/\\partial y$ [dimensionless], same shape as input.
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        sf = torch.sqrt(sf_arg)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        # d(a2*r²)/dr² = a2
        if self.ai2 is not None:
            dsdr2 = dsdr2 + self.ai2

        # Horner form of the derivative: r² * (2*a4 + r² * (3*a6 + ...)).
        if self.ai_degree > 0:
            order = self.ai_degree + 1
            polynomial = order * getattr(self, f"ai{2 * order}")
            for order in range(self.ai_degree, 1, -1):
                polynomial = order * getattr(self, f"ai{2 * order}") + r2 * polynomial
            dsdr2 = dsdr2 + r2 * polynomial

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Return a mask of points where the conic sag is real-valued.

        A point is valid when $(1+k)c^2\\rho^2 < 1$, i.e. inside the conic's
        real boundary. Fully tensorized (no Python branch on the tensor value
        of `k`) so the function is safe to trace through `torch.compile`.
        When $k \\le -1$ the conic has no real boundary, so every point is
        treated as valid.

        Args:
            x (torch.Tensor): x coordinate(s) [mm], any shape.
            y (torch.Tensor): y coordinate(s) [mm], broadcastable with `x`.

        Returns:
            valid (torch.Tensor): Boolean mask, same shape as the broadcast
                of `x` and `y`.
        """
        c, k = self._get_curvature_params()
        one_plus_k = 1 + k
        # Avoid division by zero / negative when computing the limit; the
        # bogus value is masked out by the where below.
        safe = torch.where(one_plus_k > 0, one_plus_k, torch.ones_like(one_plus_k))
        limit_sq = 1.0 / (c * c * safe)
        inside = (x * x + y * y) < limit_sq
        return torch.where(one_plus_k > 0, inside, torch.ones_like(inside))

    def max_height(self):
        """Return the maximum valid radial height of the surface.

        For an oblate/ellipsoidal conic ($k > -1$) the sag is real only up to
        $\\rho_{max} = \\sqrt{1/((k+1)c^2)}$; a small margin (0.001 mm) is
        subtracted. For $k \\le -1$ there is no boundary and a large value
        (10000 mm) is returned.

        Returns:
            max_height (float): Maximum valid radial height [mm].
        """
        c, k = self._get_curvature_params()
        if k > -1:
            return torch.sqrt(1 / (k + 1) / (c**2)).item() - 0.001
        return 10e3

    # =======================================
    # Intersection
    # =======================================

    def newton_initial_t(self, ray):
        """Seed Newton's method with the base-sphere intersection.

        The base sphere is defined by the asphere curvature ``c`` with
        conic constant and polynomial coefficients set to zero. Its vertex is
        at the local origin and its center is at ``(0, 0, 1/c)``. Of the two
        analytic intersections, this method selects the point closest to the
        vertex, matching the sag branch represented by :meth:`_sag`.

        A repeated root is handled explicitly. A flat base, a ray that misses
        the sphere, or a non-finite analytic result falls back safely to the
        vertex-plane approximation supplied by :class:`Surface`.

        Args:
            ray (Ray): Input ray bundle in local surface coordinates.

        Returns:
            t (torch.Tensor): Initial intersection parameter [mm], shape [...]
                matching the ray batch.
        """
        t_plane = super().newton_initial_t(ray)
        c = self.c

        if c.abs() < EPSILON:
            return t_plane

        # Vertex-anchored base-sphere equation:
        #
        #   c * (x^2 + y^2 + z^2) - 2z = 0.
        #
        # This form avoids the R^2 subtraction in the center-anchored
        # equation and remains accurate for shallow curvatures in float32.
        od = torch.sum(ray.o * ray.d, dim=-1)
        dd = torch.sum(ray.d * ray.d, dim=-1)
        oo = torch.sum(ray.o * ray.o, dim=-1)

        a = c * dd
        b = 2.0 * (c * od - ray.d[..., 2])
        c_coeff = c * oo - 2.0 * ray.o[..., 2]
        discriminant = b * b - 4.0 * a * c_coeff
        sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))

        # Stable quadratic roots. For a repeated root q is zero, so use the
        # standard repeated-root expression for the second candidate.
        q = torch.where(
            b >= 0,
            -(b + sqrt_discriminant) / 2.0,
            (sqrt_discriminant - b) / 2.0,
        )
        t1 = q / a
        repeated_root = -b / (2.0 * a)
        q_is_nonzero = q.abs() > EPSILON
        q_safe = torch.where(q_is_nonzero, q, torch.ones_like(q))
        t2 = torch.where(q_is_nonzero, c_coeff / q_safe, repeated_root)

        z1 = ray.o[..., 2] + t1 * ray.d[..., 2]
        z2 = ray.o[..., 2] + t2 * ray.d[..., 2]
        t_sphere = torch.where(z1.abs() < z2.abs(), t1, t2)

        has_real_finite_root = (discriminant >= 0) & torch.isfinite(t_sphere)
        return torch.where(has_real_finite_root, t_sphere, t_plane)

    # =======================================
    # Optimization
    # =======================================

    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-2, 1e-4], optim_mat=False):
        """Get optimizer parameters for different parameters.

        The learning rate for each aspheric coefficient $a_{2n}$ is scaled
        by $1 / \\max(r, 1)^{2n}$ so that the effective sag perturbation per
        Adam step is approximately constant (~lr_base mm) regardless of
        surface semi-diameter. Without this normalisation, gradients scale
        as $O(r^{2n})$ and can reach $10^5$ for camera-sized surfaces,
        causing NaN within a few dozen iterations.

        Args:
            lrs (list[float], optional): Learning rates for `[d, c, k, ai]`.
                Defaults to `[1e-4, 1e-4, 1e-2, 1e-4]`.
            optim_mat (bool, optional): Whether to also optimize the
                material parameters. Defaults to False.

        Returns:
            params (list[dict]): Parameter groups (each a dict with `params`
                and `lr`) ready to pass to a torch optimizer.
        """
        params = []

        # Optimize distance
        self.d_next.requires_grad_(True)
        params.append({"params": [self.d_next], "lr": lrs[0]})

        # Optimize curvature
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        # Optimize conic constant
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        # Optimize aspheric coefficients with r-normalised learning rates.
        # Gradient of sag w.r.t. a_{2n} scales as r^{2n}.  Dividing the lr
        # by r^{2n} keeps the effective sag change per step ≈ lr_base,
        # so every order contributes equally to surface shape evolution.
        if self.ai is not None:
            if self.ai_degree > 0:
                r_norm = max(self.r, 1.0)
                lr_base = lrs[3] if len(lrs) > 3 else 1e-4
                for i in range(self.ai_degree):
                    p_name = f"ai{2 * (i + 2)}"
                    p = getattr(self, p_name)
                    p.requires_grad_(True)
                    order = 2 * (i + 2)  # 4, 6, 8, 10, ...
                    lr_ai = lr_base / r_norm**order
                    params.append({"params": [p], "lr": lr_ai})

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Serialize the surface to a dict.

        The aspheric coefficients are written into the `ai` list as
        `[a4, a6, a8, ...]`; when the legacy `ai2` coefficient is present it
        is prepended so `ai[0] = a2` and `use_ai2` is set True.

        Returns:
            surf_dict (dict): Serialized surface (keys `type`, `r`, `roc`,
                `d`, `k`, `ai`, `use_ai2`, `mat2`, plus informational
                `(c)`/`(ai*)`/`(mat2_n)`/`(mat2_V)` entries). Lengths in [mm], `c` in [1/mm].
        """
        has_ai2 = self.ai2 is not None
        c_value = self.c.item()
        surf_dict = {
            "type": "Aspheric",
            "r": self.r,
            "(c)": c_value,
            "roc": 0.0 if c_value == 0.0 else 1.0 / c_value,
            "d_next": self.d_next.item(),
            "k": self.k.item(),
            "ai": [],
            "use_ai2": has_ai2,
            "pos_xy": [self.pos_x.item(), self.pos_y.item()],
            "vec_local": self.vec_local.tolist(),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        # Prepend a2 to ai list if present (ai2 key is informational;
        # deserialization reads ai[0] when use_ai2=True)
        if has_ai2:
            surf_dict["ai2"] = self.ai2.item()
            surf_dict["ai"].append(self.ai2.item())

        for i in range(self.ai_degree):
            order = i + 2
            coeff = getattr(self, f"ai{2 * order}")
            surf_dict[f"(ai{2 * order})"] = coeff.item()
            surf_dict["ai"].append(coeff.item())

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return the Zemax (.zmx) text block for this surface.

        Emits an `EVENASPH` surface. PARM 1 holds the legacy 2nd-order
        coefficient `a2`, and PARM 2 onward hold `a4, a6, a8, ...`, padded
        with zeros up to PARM 8 (a16).

        Args:
            surf_idx (int): Surface index in the Zemax file.
            d_next (torch.Tensor): Axial distance [mm] to the next surface.

        Returns:
            zmx_str (str): Multi-line Zemax surface description.
        """
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )

        # Collect absolute ai values, PARM 1 = a2, PARM 2+ = a4, a6, ...
        abs_ai = [self.ai2.item() if self.ai2 is not None else 0.0]
        for i in range(self.ai_degree):
            abs_ai.append(getattr(self, f"ai{2 * (i + 2)}").item())

        # Pad with zeros for Zemax PARM format (needs 8 PARMs for a2–a16)
        while len(abs_ai) < 8:
            abs_ai.append(0.0)

        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    GLAS {self.mat2.zmx_glass()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        return zmx_str

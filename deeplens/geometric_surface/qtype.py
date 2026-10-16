# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Q-type (Forbes Q-polynomial) freeform surface.

Q-type polynomials are orthogonal polynomial representations commonly used for
freeform optical surface design. This module implements the Qbfs (Q "best-fit
sphere" freeform sag) representation. All lengths are in millimetres [mm].

The total surface sag is the sum of a base conic and a Q-polynomial departure:

$$
z(x, y) = \\frac{c\\,r^2}{1 + \\sqrt{1 - (1+k)\\,c^2 r^2}}
          + u^4 \\sum_m a_m\\, Q_m^{bfs}(u^2)
$$

with $r = \\sqrt{x^2 + y^2}$, curvature $c$ [1/mm], conic constant $k$, and
normalized radial coordinate $u = r / r_{norm}$ (valid for $0 \\le u \\le 1$).

Reference:
    G. W. Forbes, "Shape specification for axially symmetric optical surfaces,"
    Opt. Express 15, 5218-5226 (2007).
    G. W. Forbes, "Robust, efficient computational methods for axially symmetric
    optical aspheres," Opt. Express 18, 19700-19712 (2010).
    ISO 10110-19:2015 - Optics and photonics - Preparation of drawings for optical
    elements and systems - Part 19: General description of surfaces and components.
"""

import numpy as np
import torch

from .base_surface import EPSILON, Surface


def compute_qbfs_polynomials(u2, n_terms):
    """Compute Qbfs polynomials $Q_0, Q_1, \\ldots, Q_{n-1}$ evaluated at $u^2$.

    The Qbfs polynomials are built from the Jacobi polynomials
    $P_m^{(0,4)}(1 - 2u^2)$ via a three-term recurrence, then rescaled by the
    Qbfs orthonormalization factor. The $(1 - u^2)^{-5/2}$ weighting of the full
    Forbes definition is NOT applied here (it is folded into the sag computation
    for numerical stability), so these are the bare normalized polynomials.

    Args:
        u2 (torch.Tensor): Squared normalized radial coordinate
            $u^2 = r^2 / r_{norm}^2$, any shape.
        n_terms (int): Number of Q-polynomial terms to compute.

    Returns:
        Q (list[torch.Tensor]): List of length `n_terms` holding
            $Q_0(u^2), Q_1(u^2), \\ldots, Q_{n\\_terms-1}(u^2)$, each the same
            shape as `u2`. Empty list when `n_terms` is 0.
    """
    if n_terms == 0:
        return []

    # Transform to Jacobi polynomial argument: x = 1 - 2*u²
    x = 1 - 2 * u2

    # Compute Jacobi polynomials P_m^(0,4)(x) using recurrence
    # P_0^(0,4)(x) = 1
    # P_1^(0,4)(x) = -2 + 3x
    # Recurrence: P_{n+1}^(0,4)(x) = (A_n * x + B_n) * P_n^(0,4)(x) - C_n * P_{n-1}^(0,4)(x)

    P = [torch.ones_like(u2)]  # P_0

    if n_terms > 1:
        P.append(-2 + 3 * x)  # P_1

    alpha, beta = 0, 4
    for n in range(1, n_terms - 1):
        # Recurrence coefficients for Jacobi polynomials
        an = 2 * n + alpha + beta
        A_n = (
            (2 * n + alpha + beta + 1)
            * (2 * n + alpha + beta + 2)
            / (2 * (n + 1) * (n + alpha + beta + 1))
        )
        B_n = (
            (alpha**2 - beta**2)
            * (2 * n + alpha + beta + 1)
            / (2 * (n + 1) * (n + alpha + beta + 1) * an)
        )
        C_n = (
            (n + alpha)
            * (n + beta)
            * (2 * n + alpha + beta + 2)
            / ((n + 1) * (n + alpha + beta + 1) * an)
        )

        P_next = (A_n * x + B_n) * P[n] - C_n * P[n - 1]
        P.append(P_next)

    # Convert to Qbfs: Q_m = P_m^(0,4)(1-2u²) * normalization / (1-u²)^(5/2)
    # The normalization ensures orthogonality
    # For numerical stability, we compute without the (1-u²)^(-5/2) factor here
    # and include it in the sag computation

    # Normalization factors for Qbfs
    # f_m = sqrt((m+1) * (m+5) * (m+2) * (m+4) * (m+3)^2 / (8 * (2m+5)))
    Q = []
    for m in range(n_terms):
        # Normalization factor
        norm = np.sqrt(
            (m + 1) * (m + 5) * (m + 2) * (m + 4) * (m + 3) ** 2 / (8 * (2 * m + 5))
        )
        # Jacobi polynomial normalization at x=1: P_m^(0,4)(1) = C(m+4, m)
        jacobi_norm = 1.0
        for k in range(1, 5):
            jacobi_norm *= (m + k) / k
        Q.append(P[m] / (jacobi_norm * norm))

    return Q


class QTypeFreeform(Surface):
    """Q-type (Forbes Qbfs polynomial) freeform surface.

    Represents a rotationally symmetric surface as a base conic plus a Forbes
    Qbfs polynomial departure. The orthogonality of the basis makes the
    coefficients well-conditioned for gradient-based optimization. Individual
    coefficients are also stored as attributes `q0`, `q1`, ... so they can be
    optimized independently. All lengths are in millimetres [mm].

    Attributes:
        c (torch.Tensor): Base-surface curvature (1 / radius of curvature) [1/mm].
        k (torch.Tensor): Conic constant.
        r_norm (float): Normalization radius for the Q polynomials [mm]; defaults
            to the aperture radius `r`.
        qm (torch.Tensor or None): Q-polynomial coefficients
            $[a_0, a_1, \\ldots, a_{n-1}]$, or None when no coefficients are set.
        n_qterms (int): Number of Q-polynomial terms (length of `qm`).
    """

    def __init__(
        self,
        r,
        d_next,
        c,
        k,
        qm,
        mat2,
        r_norm=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize a Q-type freeform surface.

        Args:
            r (float): Aperture radius (semi-diameter) of the surface [mm].
            d (float): Axial distance from the origin to the surface vertex [mm].
            c (float): Base-surface curvature (1 / radius of curvature) [1/mm].
            k (float): Conic constant (k=0 sphere, k=-1 paraboloid).
            qm (list): Q-polynomial coefficients $[a_0, a_1, \\ldots, a_{n-1}]$;
                an empty list or None gives a pure conic.
            mat2 (str or Material): Material on the exit side of the surface.
            r_norm (float, optional): Normalization radius for the Q polynomials
                [mm]. Defaults to None, in which case `r` is used.
            pos_xy (list, optional): Surface center position [x, y] [mm].
                Defaults to [0.0, 0.0].
            vec_local (list, optional): Local surface normal vector.
                Defaults to [0.0, 0.0, 1.0].
            is_square (bool, optional): Whether the aperture is square.
                Defaults to False.
            device (str, optional): Torch device. Defaults to "cpu".
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

        tensor_kwargs = {"device": device, "dtype": self.d_next.dtype}
        self.c = torch.as_tensor(c, **tensor_kwargs)
        self.k = torch.as_tensor(k, **tensor_kwargs)
        self.r_norm = r_norm if r_norm is not None else r

        # Store Q polynomial coefficients
        if qm is not None and len(qm) > 0:
            self.qm = torch.as_tensor(qm, **tensor_kwargs)
            self.n_qterms = len(qm)
            # Also store individual coefficients for optimization
            for i, coef in enumerate(qm):
                setattr(self, f"q{i}", torch.as_tensor(coef, **tensor_kwargs))
        else:
            self.qm = None
            self.n_qterms = 0

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a `QTypeFreeform` from a surface specification dictionary.

        Accepts either `roc` (radius of curvature, converted to curvature
        $c = 1 / roc$) or `c` directly. The keys `k`, `qm`, and `r_norm` are
        optional and fall back to a conic with no Q departure.

        Args:
            surf_dict (dict): Surface specification with keys `r`, `d`, `mat2`,
                and either `roc` or `c`; optional keys `k`, `qm`, `r_norm`.

        Returns:
            surface (QTypeFreeform): The constructed surface instance.
        """
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        return cls(
            r=surf_dict["r"],
            d_next=surf_dict["d_next"],
            c=c,
            k=surf_dict.get("k", 0.0),
            qm=surf_dict.get("qm", []),
            mat2=surf_dict["mat2"],
            r_norm=surf_dict.get("r_norm", None),
        )

    def paraxial_power(self, n1, n2, wvln=None):
        """Return the paraxial optical power of this surface [1/mm].

        Only the vertex curvature contributes: the Q-polynomial departure
        starts at the 4th order, so it does not affect first-order properties.

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
        """Compute the surface sag $z = f(x, y)$.

        The sag is the base conic plus the Q-polynomial departure:

        $$
        z = \\frac{c\\,r^2}{1 + \\sqrt{1 - (1+k)\\,c^2 r^2}}
            + u^4 (1 - u^2)^{5/2} \\sum_m a_m\\, Q_m(u^2)
        $$

        with $r^2 = x^2 + y^2$ and $u^2 = r^2 / r_{norm}^2$. The conic radicand
        and $1 - u^2$ are clamped to `EPSILON` to avoid NaN sag/gradients beyond
        the surface boundary.

        Args:
            x (torch.Tensor): x coordinates [mm], any shape.
            y (torch.Tensor): y coordinates [mm], same shape as `x`.

        Returns:
            z (torch.Tensor): Surface sag [mm], same shape as `x`.
        """
        c = self.c
        k = self.k

        # Radial distance squared
        r2 = x**2 + y**2

        # Base conic sag. Clamp the radicand (not + EPSILON): beyond the conic
        # boundary (1+k)c^2 r^2 > 1 the argument goes negative and torch.sqrt
        # returns NaN (and NaN gradients). Matches Aspheric._sag.
        sqrt_term = torch.sqrt(torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON))
        z_base = r2 * c / (1 + sqrt_term)

        # Q-polynomial departure
        if self.n_qterms > 0:
            # Normalized radial coordinate
            u2 = r2 / (self.r_norm**2)
            u4 = u2**2

            # Compute Q polynomials
            Q_polys = compute_qbfs_polynomials(u2, self.n_qterms)

            # Weighting factor: (1 - u²)^(5/2) for proper Qbfs behavior
            # But for numerical stability near u=1, we use a soft clamp
            one_minus_u2 = torch.clamp(1 - u2, min=EPSILON)
            weight = one_minus_u2 ** (5 / 2)

            # Sum Q polynomial contributions
            z_q = torch.zeros_like(x)
            for m in range(self.n_qterms):
                qm_coef = getattr(self, f"q{m}")
                z_q = z_q + qm_coef * Q_polys[m]

            # Apply u⁴ factor and weight
            z_q = u4 * weight * z_q

            return z_base + z_q

        return z_base

    def _dfdxy(self, x, y):
        """Compute first-order sag derivatives w.r.t. $x$ and $y$.

        Applies the chain rule through $r^2$: $\\partial z/\\partial x =
        (\\partial z/\\partial r^2)\\,(2x)$. The base-conic part is differentiated
        analytically; the Q-polynomial derivative $\\partial Q_m/\\partial u^2$ is
        approximated by a forward finite difference (step $10^{-7}$).

        Args:
            x (torch.Tensor): x coordinates [mm], any shape.
            y (torch.Tensor): y coordinates [mm], same shape as `x`.

        Returns:
            dfdx (torch.Tensor): $\\partial z/\\partial x$ [dimensionless], same
                shape as `x`.
            dfdy (torch.Tensor): $\\partial z/\\partial y$ [dimensionless], same
                shape as `x`.
        """
        c = self.c
        k = self.k

        r2 = x**2 + y**2

        # Base conic derivative dz_base/dr². Clamp the radicand to keep it
        # non-negative (see _sag); + EPSILON does not prevent a NaN sqrt.
        sqrt_term = torch.sqrt(torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON))
        dz_base_dr2 = (
            c
            * (1 + sqrt_term + (1 + k) * r2 * c**2 / (2 * sqrt_term))
            / (1 + sqrt_term) ** 2
        )

        # Q-polynomial derivative
        if self.n_qterms > 0:
            u2 = r2 / (self.r_norm**2)
            u4 = u2**2

            # Compute Q polynomials and their derivatives
            Q_polys = compute_qbfs_polynomials(u2, self.n_qterms)

            # Weight factor
            one_minus_u2 = torch.clamp(1 - u2, min=EPSILON)
            weight = one_minus_u2 ** (5 / 2)

            # d(weight)/du² = (5/2) * (1-u²)^(3/2) * (-1) = -(5/2) * (1-u²)^(3/2)
            dweight_du2 = -2.5 * one_minus_u2 ** (3 / 2)

            # Sum and derivatives
            Q_sum = torch.zeros_like(x)
            dQ_sum_du2 = torch.zeros_like(x)

            # For derivative of Q polynomials, use finite difference for now
            delta = 1e-7
            Q_polys_plus = compute_qbfs_polynomials(u2 + delta, self.n_qterms)

            for m in range(self.n_qterms):
                qm_coef = getattr(self, f"q{m}")
                Q_sum = Q_sum + qm_coef * Q_polys[m]
                dQ_du2 = (Q_polys_plus[m] - Q_polys[m]) / delta
                dQ_sum_du2 = dQ_sum_du2 + qm_coef * dQ_du2

            # z_q = u⁴ * weight * Q_sum
            # dz_q/du² = 2u² * weight * Q_sum + u⁴ * dweight/du² * Q_sum + u⁴ * weight * dQ_sum/du²
            dz_q_du2 = (
                2 * u2 * weight * Q_sum
                + u4 * dweight_du2 * Q_sum
                + u4 * weight * dQ_sum_du2
            )

            # Convert du²/dr² = 1/r_norm²
            dz_q_dr2 = dz_q_du2 / (self.r_norm**2)

            dz_dr2 = dz_base_dr2 + dz_q_dr2
        else:
            dz_dr2 = dz_base_dr2

        # Chain rule: dz/dx = dz/dr² * 2x, dz/dy = dz/dr² * 2y
        return dz_dr2 * 2 * x, dz_dr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Check whether points lie within the valid surface data range.

        A point is valid when it is inside the conic boundary (only present when
        $k > -1$ and $c \\ne 0$) AND within the Q-polynomial normalization radius
        ($u^2 \\le 1$). The conic test is fully tensorized so it is safe under
        `torch.compile`.

        Args:
            x (torch.Tensor): x coordinates [mm], any shape.
            y (torch.Tensor): y coordinates [mm], same shape as `x`.

        Returns:
            mask (torch.Tensor): Boolean tensor, True where the point is valid,
                same shape as `x`.
        """
        c = self.c
        k = self.k

        r2 = x**2 + y**2

        # Check conic validity. Fully tensorized (no Python branch on the
        # tensor values of k/c) so the function is safe under torch.compile.
        # A real conic boundary exists only when k > -1 AND c != 0; otherwise
        # every point is treated as valid (mirrors Aspheric.is_within_data_range).
        has_boundary = (1 + k > 0) & (c.abs() > EPSILON)
        one_plus_k = 1 + k
        c2 = c * c
        # Avoid div-by-zero / negative when the boundary is absent; the bogus
        # value is masked out by the where below.
        denom = torch.where(
            has_boundary, c2 * one_plus_k, torch.ones_like(c2 * one_plus_k)
        )
        limit_sq = 1.0 / denom - EPSILON
        inside = r2 < limit_sq
        valid_conic = torch.where(has_boundary, inside, torch.ones_like(inside))

        # Check normalized radius (should be <= 1 for Q polynomials)
        u2 = r2 / (self.r_norm**2)
        valid_qpoly = u2 <= 1 + EPSILON

        return valid_conic & valid_qpoly

    def max_height(self):
        """Return the maximum valid radial height of the surface.

        Takes the smaller of the conic boundary radius (when $k > -1$ and
        $c \\ne 0$, else a large fallback of 1e4) and the Q-polynomial
        normalization radius `r_norm`.

        Returns:
            height (float): Maximum valid radial distance [mm].
        """
        c = self.c
        k = self.k

        # Conic limit
        if k > -1 and abs(c) > EPSILON:
            max_conic = np.sqrt(1 / ((k + 1) * c**2)) - 0.001
        else:
            max_conic = 10e3

        # Q polynomial limit (normalization radius)
        max_q = self.r_norm

        return min(max_conic, max_q)

    # =======================================
    # Optimization
    # =======================================

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-2, 1e-6], decay=0.1, optim_mat=False
    ):
        """Build per-parameter optimizer groups for this surface.

        Enables gradients on `d`, `c`, `k`, and each Q coefficient. The learning
        rate for the m-th Q coefficient is `lrs[3] * decay**m`, so higher-order
        terms are tuned more slowly.

        Args:
            lrs (list, optional): Learning rates for [d, c, k, q_coefficients].
                Defaults to [1e-4, 1e-4, 1e-2, 1e-6].
            decay (float, optional): Per-order decay applied to the Q-coefficient
                learning rate. Defaults to 0.1.
            optim_mat (bool, optional): Whether to also optimize the exit-side
                material. Defaults to False.

        Returns:
            params (list): List of optimizer parameter-group dicts, each with
                keys "params" and "lr".
        """
        params = []

        # Distance
        self.d_next.requires_grad_(True)
        params.append({"params": [self.d_next], "lr": lrs[0]})

        # Curvature
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        # Conic constant
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        # Q polynomial coefficients
        if self.n_qterms > 0:
            base_lr = lrs[3] if len(lrs) > 3 else 1e-6
            for m in range(self.n_qterms):
                qm = getattr(self, f"q{m}")
                qm.requires_grad_(True)
                # Decay learning rate for higher order terms
                lr = base_lr * (decay**m)
                params.append({"params": [qm], "lr": lr})

        # Material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # IO
    # =======================================

    def surf_dict(self):
        """Serialize the surface to a dictionary.

        Includes the type tag, geometry (`r`, `d`, `c`, `roc`, `k`, `r_norm`),
        the Q-coefficient list `qm`, and the exit material name. Display-only
        keys such as `(c)` and `(q0)` carry rounded scalar values.

        Returns:
            surf_dict (dict): Dictionary representation of the surface.
        """
        surf_dict = {
            "type": "QTypeFreeform",
            "r": round(self.r, 4),
            "d_next": round(self.d_next.item(), 4),
            "(c)": round(self.c.item(), 6),
            "roc": round(1 / self.c.item(), 4)
            if abs(self.c.item()) > EPSILON
            else float("inf"),
            "k": round(self.k.item(), 6),
            "r_norm": round(self.r_norm, 4),
            "qm": [],
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            surf_dict["qm"].append(float(format(qm.item(), ".6e")))
            surf_dict[f"(q{m})"] = float(format(qm.item(), ".6e"))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return the Zemax (.zmx) surface description string.

        Emits a `QTYPE` surface with curvature, thickness, aperture, conic, the
        normalization radius (PARM 1), and the Q coefficients (PARM 2, 3, ...).

        Args:
            surf_idx (int): Surface index in the Zemax file.
            d_next (torch.Tensor): Distance to the next surface [mm] (scalar).

        Returns:
            zmx_str (str): Multi-line Zemax surface description.

        Note:
            Zemax's QTYPE representation differs from this implementation, so the
            export is approximate and may need adjustment for specific versions.
        """
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE QTYPE
    CURV {self.c.item()}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k.item()}
    PARM 1 {self.r_norm}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE QTYPE
    CURV {self.c.item()}
    DISZ {d_next.item()}
    GLAS {self.mat2.zmx_glass()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k.item()}
    PARM 1 {self.r_norm}
"""

        # Add Q coefficients
        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            zmx_str += f"    PARM {m + 2} {qm.item()}\n"

        return zmx_str

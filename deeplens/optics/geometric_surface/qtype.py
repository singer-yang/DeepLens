# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Q-type (Forbes Q-polynomial) freeform surface.

Q-type polynomials are orthogonal polynomial representations commonly used for 
freeform optical surface design. This module implements Qbfs (Q-basic freeform sag)
surface representation.

The surface sag is given by:
    z(x, y) = z_base(x, y) + z_Qbfs(x, y)

Where:
    z_base = c * r² / (1 + √(1 - (1+k) * c² * r²))  (standard conic)
    z_Qbfs = u⁴ * Σ aₘ * Qₘ^bfs(u²)                 (Q-polynomial departure)
    
    r = √(x² + y²)
    u = r / r_norm  (normalized radial coordinate, 0 ≤ u ≤ 1)

References:
    [1] G. W. Forbes, "Shape specification for axially symmetric optical surfaces," 
        Opt. Express 15, 5218-5226 (2007).
    [2] G. W. Forbes, "Robust, efficient computational methods for axially symmetric 
        optical aspheres," Opt. Express 18, 19700-19712 (2010).
    [3] ISO 10110-19:2015 - Optics and photonics - Preparation of drawings for optical 
        elements and systems - Part 19: General description of surfaces and components.
"""

import numpy as np
import torch

from deeplens.optics.geometric_surface.base import EPSILON, Surface


def compute_qbfs_polynomials(u2, n_terms):
    """Compute Qbfs polynomials Q_0, Q_1, ..., Q_{n_terms-1} at u².
    
    The Qbfs polynomials are defined by the recurrence relation:
        Q_0(u²) = 1
        Q_1(u²) = (1 - (13/5) * u²) / (1 - u²)^(5/2)  [normalized form]
        
    Using the Jacobi polynomial representation:
        Q_m^bfs(u²) = P_m^(0,4)(1 - 2u²) / (1 - u²)^(5/2) * normalization
        
    Args:
        u2: Squared normalized radial coordinate (u² = r²/r_norm²), tensor
        n_terms: Number of Q polynomial terms to compute
        
    Returns:
        List of tensors containing Q_0(u²), Q_1(u²), ..., Q_{n_terms-1}(u²)
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
        A_n = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2) / (2 * (n + 1) * (n + alpha + beta + 1))
        B_n = (alpha**2 - beta**2) * (2 * n + alpha + beta + 1) / (2 * (n + 1) * (n + alpha + beta + 1) * an)
        C_n = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2) / ((n + 1) * (n + alpha + beta + 1) * an)
        
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
        norm = np.sqrt((m + 1) * (m + 5) * (m + 2) * (m + 4) * (m + 3)**2 / (8 * (2 * m + 5)))
        # Jacobi polynomial normalization at x=1: P_m^(0,4)(1) = C(m+4, m)
        jacobi_norm = 1.0
        for k in range(1, 5):
            jacobi_norm *= (m + k) / k
        Q.append(P[m] / (jacobi_norm * norm))
    
    return Q


class QTypeFreeform(Surface):
    """Q-type (Forbes Qbfs polynomial) freeform surface.
    
    This surface type uses Forbes Q-polynomials to represent rotationally symmetric
    aspheric departures from a base conic surface. The representation is well-suited
    for optimization due to the orthogonality of the basis functions.
    
    Attributes:
        c (tensor): Curvature of the base surface (1/radius of curvature)
        k (tensor): Conic constant
        r_norm (float): Normalization radius for Q polynomials (typically equals r)
        qm (tensor): Q polynomial coefficients [a_0, a_1, ..., a_{n-1}]
    """
    
    def __init__(
        self,
        r,
        d,
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
        """Initialize Q-type freeform surface.
        
        Args:
            r (float): Aperture radius of the surface
            d (float): Distance from origin to surface vertex
            c (float): Curvature of the base surface (1/radius of curvature)
            k (float): Conic constant (k=0 for sphere, k=-1 for paraboloid)
            qm (list): Q polynomial coefficients [a_0, a_1, ..., a_{n-1}]
            mat2 (str or Material): Material after the surface
            r_norm (float, optional): Normalization radius. Defaults to r.
            pos_xy (list): Surface center position [x, y]. Defaults to [0, 0].
            vec_local (list): Local surface normal. Defaults to [0, 0, 1].
            is_square (bool): Whether aperture is square. Defaults to False.
            device (str): Torch device. Defaults to "cpu".
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
        self.r_norm = r_norm if r_norm is not None else r
        
        # Store Q polynomial coefficients
        if qm is not None and len(qm) > 0:
            self.qm = torch.tensor(qm, dtype=torch.float64)
            self.n_qterms = len(qm)
            # Also store individual coefficients for optimization
            for i, coef in enumerate(qm):
                setattr(self, f"q{i}", torch.tensor(coef))
        else:
            self.qm = None
            self.n_qterms = 0
        
        self.tolerancing = False
        self.to(device)
    
    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize surface from a dictionary specification."""
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]
        
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict.get("k", 0.0),
            qm=surf_dict.get("qm", []),
            mat2=surf_dict["mat2"],
            r_norm=surf_dict.get("r_norm", None),
        )
    
    def _sag(self, x, y):
        """Compute surface sag z = f(x, y).
        
        The sag consists of:
        1. Base conic sag: c*r² / (1 + √(1 - (1+k)*c²*r²))
        2. Q-polynomial departure: u⁴ * Σ aₘ * Qₘ(u²)
        
        Args:
            x (tensor): x coordinates
            y (tensor): y coordinates
            
        Returns:
            tensor: Surface sag values
        """
        # Tolerance handling
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k
        
        # Radial distance squared
        r2 = x**2 + y**2
        
        # Base conic sag
        sqrt_term = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)
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
            weight = one_minus_u2**(5/2)
            
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
        """Compute first-order derivatives of sag with respect to x and y.
        
        Uses chain rule: dz/dx = dz/dr² * dr²/dx = dz/dr² * 2x
        
        Args:
            x (tensor): x coordinates
            y (tensor): y coordinates
            
        Returns:
            tuple: (dz/dx, dz/dy)
        """
        # Tolerance handling
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k
        
        r2 = x**2 + y**2
        
        # Base conic derivative dz_base/dr²
        sqrt_term = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)
        dz_base_dr2 = c * (1 + sqrt_term + (1 + k) * r2 * c**2 / (2 * sqrt_term)) / (1 + sqrt_term)**2
        
        # Q-polynomial derivative
        if self.n_qterms > 0:
            u2 = r2 / (self.r_norm**2)
            u4 = u2**2
            
            # Compute Q polynomials and their derivatives
            Q_polys = compute_qbfs_polynomials(u2, self.n_qterms)
            
            # Weight factor
            one_minus_u2 = torch.clamp(1 - u2, min=EPSILON)
            weight = one_minus_u2**(5/2)
            
            # d(weight)/du² = (5/2) * (1-u²)^(3/2) * (-1) = -(5/2) * (1-u²)^(3/2)
            dweight_du2 = -2.5 * one_minus_u2**(3/2)
            
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
            dz_q_du2 = (2 * u2 * weight * Q_sum + 
                        u4 * dweight_du2 * Q_sum + 
                        u4 * weight * dQ_sum_du2)
            
            # Convert du²/dr² = 1/r_norm²
            dz_q_dr2 = dz_q_du2 / (self.r_norm**2)
            
            dz_dr2 = dz_base_dr2 + dz_q_dr2
        else:
            dz_dr2 = dz_base_dr2
        
        # Chain rule: dz/dx = dz/dr² * 2x, dz/dy = dz/dr² * 2y
        return dz_dr2 * 2 * x, dz_dr2 * 2 * y
    
    def is_within_data_range(self, x, y):
        """Check if points are within valid surface data range.
        
        For conic surfaces with k > -1, there's a maximum valid radius.
        
        Args:
            x (tensor): x coordinates
            y (tensor): y coordinates
            
        Returns:
            tensor: Boolean mask of valid points
        """
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
            c = self.c
            k = self.k
        
        r2 = x**2 + y**2
        
        # Check conic validity
        if k > -1 and abs(c) > EPSILON:
            valid_conic = r2 < 1 / (c**2 * (1 + k)) - EPSILON
        else:
            valid_conic = torch.ones_like(x, dtype=torch.bool)
        
        # Check normalized radius (should be <= 1 for Q polynomials)
        u2 = r2 / (self.r_norm**2)
        valid_qpoly = u2 <= 1 + EPSILON
        
        return valid_conic & valid_qpoly
    
    def max_height(self):
        """Maximum valid height (radial distance)."""
        if self.tolerancing:
            c = self.c + self.c_error
            k = self.k + self.k_error
        else:
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
    
    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-2, 1e-6], decay=0.1, optim_mat=False):
        """Get optimizer parameters for different surface parameters.
        
        Args:
            lrs (list): Learning rates for [d, c, k, q_coefficients].
            decay (float): Decay factor for higher-order Q coefficients.
            optim_mat (bool): Whether to optimize material parameters.
            
        Returns:
            list: Parameter groups for optimizer
        """
        params = []
        
        # Distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})
        
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
                lr = base_lr * (decay ** m)
                params.append({"params": [qm], "lr": lr})
        
        # Material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()
        
        return params
    
    # =======================================
    # Tolerancing
    # =======================================
    
    @torch.no_grad()
    def init_tolerance(self, tolerance_params=None):
        """Initialize tolerance parameters for manufacturing error simulation."""
        super().init_tolerance(tolerance_params)
        if tolerance_params is None:
            tolerance_params = {}
        self.c_tole = tolerance_params.get("c_tole", 0.001)
        self.k_tole = tolerance_params.get("k_tole", 0.001)
    
    def sample_tolerance(self):
        """Sample random manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)
    
    def zero_tolerance(self):
        """Reset all tolerances to zero."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0
    
    # =======================================
    # IO
    # =======================================
    
    def surf_dict(self):
        """Return dictionary representation of surface."""
        surf_dict = {
            "type": "QTypeFreeform",
            "r": round(self.r, 4),
            "d": round(self.d.item(), 4),
            "(c)": round(self.c.item(), 6),
            "roc": round(1 / self.c.item(), 4) if abs(self.c.item()) > EPSILON else float('inf'),
            "k": round(self.k.item(), 6),
            "r_norm": round(self.r_norm, 4),
            "qm": [],
            "mat2": self.mat2.get_name(),
        }
        
        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            surf_dict["qm"].append(float(format(qm.item(), ".6e")))
            surf_dict[f"(q{m})"] = float(format(qm.item(), ".6e"))
        
        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string.
        
        Note: Zemax uses a different Q-type representation (QTYPE surface).
        This export is approximate and may need adjustment for specific Zemax versions.
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
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k.item()}
    PARM 1 {self.r_norm}
"""
        
        # Add Q coefficients
        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            zmx_str += f"    PARM {m + 2} {qm.item()}\n"
        
        return zmx_str


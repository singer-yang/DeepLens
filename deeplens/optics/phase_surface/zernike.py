"""Zernike phase on a plane surface."""

import math

import torch

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class ZernikePhase(Phase):
    """Zernike phase on a plane surface.

    This class implements a diffractive surface using Zernike polynomials
    to represent the phase profile. It supports up to 37 Zernike terms.
    Inherits core ray-tracing functionality from Phase class.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
        zernike_order=37,
        zernike_coeff=None,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=False,
        device="cpu",
    ):
        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]
        """Initialize Zernike phase surface.

        Args:
            r: Radius of the surface
            d: Distance to next surface
            zernike_order: Number of Zernike terms (default: 37)
            norm_radii: Normalization radius (default: r)
            mat2: Material on the right side (default: "air")
            pos_xy: Position in xy plane
            vec_local: Local coordinate system vector
            is_square: Whether the aperture is square
            device: Computation device
        """
        # Initialize parent Phase class but skip param_model initialization
        # We'll set up Zernike-specific parameters manually
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # Override param_model to "zernike"
        self.param_model = "zernike"

        # Zernike polynomial parameterization
        self.zernike_order = zernike_order
        if zernike_coeff is None:
            self.z_coeff = torch.randn(self.zernike_order) * 1e-3
        else:
            self.z_coeff = torch.tensor(zernike_coeff)

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize Zernike phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        zernike_order = surf_dict.get("zernike_order", 37)

        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            zernike_order=zernike_order,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # Load Zernike coefficients
        z_coeff = surf_dict.get("z_coeff", None)
        if z_coeff is not None:
            obj.z_coeff = (
                torch.tensor(z_coeff, device=obj.device)
                if not isinstance(z_coeff, torch.Tensor)
                else z_coeff.to(obj.device)
            )

        return obj

    # ==============================
    # Zernike-specific Phase Methods
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength using Zernike polynomials.

        Overrides the parent Phase.phi() method to use Zernike polynomial representation.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        phi = self._calculate_zernike_phase(x_norm, y_norm)
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points.

        Overrides the parent Phase.dphi_dxy() method to use Zernike derivatives.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        dphidx, dphidy = self._calculate_zernike_derivatives(x_norm, y_norm)
        return dphidx, dphidy

    # ==============================
    # Zernike Polynomial Calculations
    # ==============================
    def _calculate_zernike_phase(self, x_norm, y_norm):
        """Calculate phase map using Zernike polynomials.

        Args:
            x_norm: Normalized x coordinates (range -1 to 1)
            y_norm: Normalized y coordinates (range -1 to 1)

        Returns:
            Phase map in radians
        """
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)
        alpha = torch.atan2(y_norm, x_norm)

        # Calculate all 37 Zernike terms
        Z1 = self.z_coeff[0] * 1  # piston
        Z2 = self.z_coeff[1] * 2 * r * torch.sin(alpha)  # tip/tilt
        Z3 = self.z_coeff[2] * 2 * r * torch.cos(alpha)  # tip/tilt
        Z4 = self.z_coeff[3] * math.sqrt(3) * (2 * r**2 - 1)  # defocus
        Z5 = self.z_coeff[4] * math.sqrt(6) * r**2 * torch.sin(2 * alpha)
        Z6 = self.z_coeff[5] * math.sqrt(6) * r**2 * torch.cos(2 * alpha)
        Z7 = self.z_coeff[6] * math.sqrt(8) * (3 * r**3 - 2 * r) * torch.sin(alpha)
        Z8 = self.z_coeff[7] * math.sqrt(8) * (3 * r**3 - 2 * r) * torch.cos(alpha)
        Z9 = self.z_coeff[8] * math.sqrt(8) * r**3 * torch.sin(3 * alpha)
        Z10 = self.z_coeff[9] * math.sqrt(8) * r**3 * torch.cos(3 * alpha)
        Z11 = self.z_coeff[10] * math.sqrt(5) * (6 * r**4 - 6 * r**2 + 1)
        Z12 = (
            self.z_coeff[11]
            * math.sqrt(10)
            * (4 * r**4 - 3 * r**2)
            * torch.cos(2 * alpha)
        )
        Z13 = (
            self.z_coeff[12]
            * math.sqrt(10)
            * (4 * r**4 - 3 * r**2)
            * torch.sin(2 * alpha)
        )
        Z14 = self.z_coeff[13] * math.sqrt(10) * r**4 * torch.cos(4 * alpha)
        Z15 = self.z_coeff[14] * math.sqrt(10) * r**4 * torch.sin(4 * alpha)
        Z16 = (
            self.z_coeff[15]
            * math.sqrt(12)
            * (10 * r**5 - 12 * r**3 + 3 * r)
            * torch.cos(alpha)
        )
        Z17 = (
            self.z_coeff[16]
            * math.sqrt(12)
            * (10 * r**5 - 12 * r**3 + 3 * r)
            * torch.sin(alpha)
        )
        Z18 = (
            self.z_coeff[17]
            * math.sqrt(12)
            * (5 * r**5 - 4 * r**3)
            * torch.cos(3 * alpha)
        )
        Z19 = (
            self.z_coeff[18]
            * math.sqrt(12)
            * (5 * r**5 - 4 * r**3)
            * torch.sin(3 * alpha)
        )
        Z20 = self.z_coeff[19] * math.sqrt(12) * r**5 * torch.cos(5 * alpha)
        Z21 = self.z_coeff[20] * math.sqrt(12) * r**5 * torch.sin(5 * alpha)
        Z22 = self.z_coeff[21] * math.sqrt(7) * (20 * r**6 - 30 * r**4 + 12 * r**2 - 1)
        Z23 = (
            self.z_coeff[22]
            * math.sqrt(14)
            * (15 * r**6 - 20 * r**4 + 6 * r**2)
            * torch.sin(2 * alpha)
        )
        Z24 = (
            self.z_coeff[23]
            * math.sqrt(14)
            * (15 * r**6 - 20 * r**4 + 6 * r**2)
            * torch.cos(2 * alpha)
        )
        Z25 = (
            self.z_coeff[24]
            * math.sqrt(14)
            * (6 * r**6 - 5 * r**4)
            * torch.sin(4 * alpha)
        )
        Z26 = (
            self.z_coeff[25]
            * math.sqrt(14)
            * (6 * r**6 - 5 * r**4)
            * torch.cos(4 * alpha)
        )
        Z27 = self.z_coeff[26] * math.sqrt(14) * r**6 * torch.sin(6 * alpha)
        Z28 = self.z_coeff[27] * math.sqrt(14) * r**6 * torch.cos(6 * alpha)
        Z29 = (
            self.z_coeff[28]
            * 4
            * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r)
            * torch.sin(alpha)
        )
        Z30 = (
            self.z_coeff[29]
            * 4
            * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r)
            * torch.cos(alpha)
        )
        Z31 = (
            self.z_coeff[30]
            * 4
            * (21 * r**7 - 30 * r**5 + 10 * r**3)
            * torch.sin(3 * alpha)
        )
        Z32 = (
            self.z_coeff[31]
            * 4
            * (21 * r**7 - 30 * r**5 + 10 * r**3)
            * torch.cos(3 * alpha)
        )
        Z33 = self.z_coeff[32] * 4 * (7 * r**7 - 6 * r**5) * torch.sin(5 * alpha)
        Z34 = self.z_coeff[33] * 4 * (7 * r**7 - 6 * r**5) * torch.cos(5 * alpha)
        Z35 = self.z_coeff[34] * 4 * r**7 * torch.sin(7 * alpha)
        Z36 = self.z_coeff[35] * 4 * r**7 * torch.cos(7 * alpha)
        Z37 = (
            self.z_coeff[36] * 3 * (70 * r**8 - 140 * r**6 + 90 * r**4 - 20 * r**2 + 1)
        )

        # Sum all Zernike terms
        ZW = (
            Z1
            + Z2
            + Z3
            + Z4
            + Z5
            + Z6
            + Z7
            + Z8
            + Z9
            + Z10
            + Z11
            + Z12
            + Z13
            + Z14
            + Z15
            + Z16
            + Z17
            + Z18
            + Z19
            + Z20
            + Z21
            + Z22
            + Z23
            + Z24
            + Z25
            + Z26
            + Z27
            + Z28
            + Z29
            + Z30
            + Z31
            + Z32
            + Z33
            + Z34
            + Z35
            + Z36
            + Z37
        )

        # Apply circular mask (set phase to 0 outside unit circle)
        mask = torch.gt(x_norm**2 + y_norm**2, 1)
        ZW = torch.where(mask, torch.zeros_like(ZW), ZW)

        return ZW

    def _calculate_zernike_derivatives(self, x_norm, y_norm):
        """Calculate derivatives of Zernike phase with respect to x and y.

        Args:
            x_norm: Normalized x coordinates (range -1 to 1)
            y_norm: Normalized y coordinates (range -1 to 1)

        Returns:
            dphidx, dphidy: Phase derivatives in x and y directions
        """
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)
        alpha = torch.atan2(y_norm, x_norm)

        # Precompute trigonometric functions
        sin_a = torch.sin(alpha)
        cos_a = torch.cos(alpha)
        sin_2a = torch.sin(2 * alpha)
        cos_2a = torch.cos(2 * alpha)
        sin_3a = torch.sin(3 * alpha)
        cos_3a = torch.cos(3 * alpha)
        sin_4a = torch.sin(4 * alpha)
        cos_4a = torch.cos(4 * alpha)
        sin_5a = torch.sin(5 * alpha)
        cos_5a = torch.cos(5 * alpha)
        sin_6a = torch.sin(6 * alpha)
        cos_6a = torch.cos(6 * alpha)
        sin_7a = torch.sin(7 * alpha)
        cos_7a = torch.cos(7 * alpha)

        # For each Zernike term Z = coeff * R(r) * T(theta), we have:
        # dZ/dx = coeff * (dR/dr * dr/dx * T + R * dT/dtheta * dtheta/dx)
        # dZ/dy = coeff * (dR/dr * dr/dy * T + R * dT/dtheta * dtheta/dy)
        # where dr/dx = x/r, dr/dy = y/r, dtheta/dx = -y/r^2, dtheta/dy = x/r^2

        # Handle dr/dx, dr/dy, dtheta/dx, dtheta/dy
        drdx = x_norm / r
        drdy = y_norm / r
        dthetadx = -y_norm / (r**2 + EPSILON)
        dthetady = x_norm / (r**2 + EPSILON)

        # Initialize derivatives
        dZdx = torch.zeros_like(x_norm)
        dZdy = torch.zeros_like(y_norm)

        # Z1 = c1 * 1 (piston - no derivative)
        # dZdx += 0, dZdy += 0

        # Z2 = c2 * 2*r*sin(a)
        R2 = 2 * r
        dR2dr = 2
        T2 = sin_a
        dT2dtheta = cos_a
        dZdx += self.z_coeff[1] * (dR2dr * drdx * T2 + R2 * dT2dtheta * dthetadx)
        dZdy += self.z_coeff[1] * (dR2dr * drdy * T2 + R2 * dT2dtheta * dthetady)

        # Z3 = c3 * 2*r*cos(a)
        R3 = 2 * r
        dR3dr = 2
        T3 = cos_a
        dT3dtheta = -sin_a
        dZdx += self.z_coeff[2] * (dR3dr * drdx * T3 + R3 * dT3dtheta * dthetadx)
        dZdy += self.z_coeff[2] * (dR3dr * drdy * T3 + R3 * dT3dtheta * dthetady)

        # Z4 = c4 * sqrt(3) * (2*r^2 - 1)
        dR4dr = math.sqrt(3) * 4 * r
        dZdx += self.z_coeff[3] * dR4dr * drdx
        dZdy += self.z_coeff[3] * dR4dr * drdy

        # Z5 = c5 * sqrt(6) * r^2 * sin(2*a)
        R5 = math.sqrt(6) * r**2
        dR5dr = math.sqrt(6) * 2 * r
        T5 = sin_2a
        dT5dtheta = 2 * cos_2a
        dZdx += self.z_coeff[4] * (dR5dr * drdx * T5 + R5 * dT5dtheta * dthetadx)
        dZdy += self.z_coeff[4] * (dR5dr * drdy * T5 + R5 * dT5dtheta * dthetady)

        # Z6 = c6 * sqrt(6) * r^2 * cos(2*a)
        R6 = math.sqrt(6) * r**2
        dR6dr = math.sqrt(6) * 2 * r
        T6 = cos_2a
        dT6dtheta = -2 * sin_2a
        dZdx += self.z_coeff[5] * (dR6dr * drdx * T6 + R6 * dT6dtheta * dthetadx)
        dZdy += self.z_coeff[5] * (dR6dr * drdy * T6 + R6 * dT6dtheta * dthetady)

        # Z7 = c7 * sqrt(8) * (3*r^3 - 2*r) * sin(a)
        R7 = math.sqrt(8) * (3 * r**3 - 2 * r)
        dR7dr = math.sqrt(8) * (9 * r**2 - 2)
        T7 = sin_a
        dT7dtheta = cos_a
        dZdx += self.z_coeff[6] * (dR7dr * drdx * T7 + R7 * dT7dtheta * dthetadx)
        dZdy += self.z_coeff[6] * (dR7dr * drdy * T7 + R7 * dT7dtheta * dthetady)

        # Z8 = c8 * sqrt(8) * (3*r^3 - 2*r) * cos(a)
        R8 = math.sqrt(8) * (3 * r**3 - 2 * r)
        dR8dr = math.sqrt(8) * (9 * r**2 - 2)
        T8 = cos_a
        dT8dtheta = -sin_a
        dZdx += self.z_coeff[7] * (dR8dr * drdx * T8 + R8 * dT8dtheta * dthetadx)
        dZdy += self.z_coeff[7] * (dR8dr * drdy * T8 + R8 * dT8dtheta * dthetady)

        # Z9 = c9 * sqrt(8) * r^3 * sin(3*a)
        R9 = math.sqrt(8) * r**3
        dR9dr = math.sqrt(8) * 3 * r**2
        T9 = sin_3a
        dT9dtheta = 3 * cos_3a
        dZdx += self.z_coeff[8] * (dR9dr * drdx * T9 + R9 * dT9dtheta * dthetadx)
        dZdy += self.z_coeff[8] * (dR9dr * drdy * T9 + R9 * dT9dtheta * dthetady)

        # Z10 = c10 * sqrt(8) * r^3 * cos(3*a)
        R10 = math.sqrt(8) * r**3
        dR10dr = math.sqrt(8) * 3 * r**2
        T10 = cos_3a
        dT10dtheta = -3 * sin_3a
        dZdx += self.z_coeff[9] * (dR10dr * drdx * T10 + R10 * dT10dtheta * dthetadx)
        dZdy += self.z_coeff[9] * (dR10dr * drdy * T10 + R10 * dT10dtheta * dthetady)

        # Continue with higher order terms...
        # Z11 = c11 * sqrt(5) * (6*r^4 - 6*r^2 + 1)
        dR11dr = math.sqrt(5) * (24 * r**3 - 12 * r)
        dZdx += self.z_coeff[10] * dR11dr * drdx
        dZdy += self.z_coeff[10] * dR11dr * drdy

        # Z12 = c12 * sqrt(10) * (4*r^4 - 3*r^2) * cos(2*a)
        R12 = math.sqrt(10) * (4 * r**4 - 3 * r**2)
        dR12dr = math.sqrt(10) * (16 * r**3 - 6 * r)
        T12 = cos_2a
        dT12dtheta = -2 * sin_2a
        dZdx += self.z_coeff[11] * (dR12dr * drdx * T12 + R12 * dT12dtheta * dthetadx)
        dZdy += self.z_coeff[11] * (dR12dr * drdy * T12 + R12 * dT12dtheta * dthetady)

        # Z13 = c13 * sqrt(10) * (4*r^4 - 3*r^2) * sin(2*a)
        R13 = math.sqrt(10) * (4 * r**4 - 3 * r**2)
        dR13dr = math.sqrt(10) * (16 * r**3 - 6 * r)
        T13 = sin_2a
        dT13dtheta = 2 * cos_2a
        dZdx += self.z_coeff[12] * (dR13dr * drdx * T13 + R13 * dT13dtheta * dthetadx)
        dZdy += self.z_coeff[12] * (dR13dr * drdy * T13 + R13 * dT13dtheta * dthetady)

        # Z14 = c14 * sqrt(10) * r^4 * cos(4*a)
        R14 = math.sqrt(10) * r**4
        dR14dr = math.sqrt(10) * 4 * r**3
        T14 = cos_4a
        dT14dtheta = -4 * sin_4a
        dZdx += self.z_coeff[13] * (dR14dr * drdx * T14 + R14 * dT14dtheta * dthetadx)
        dZdy += self.z_coeff[13] * (dR14dr * drdy * T14 + R14 * dT14dtheta * dthetady)

        # Z15 = c15 * sqrt(10) * r^4 * sin(4*a)
        R15 = math.sqrt(10) * r**4
        dR15dr = math.sqrt(10) * 4 * r**3
        T15 = sin_4a
        dT15dtheta = 4 * cos_4a
        dZdx += self.z_coeff[14] * (dR15dr * drdx * T15 + R15 * dT15dtheta * dthetadx)
        dZdy += self.z_coeff[14] * (dR15dr * drdy * T15 + R15 * dT15dtheta * dthetady)

        # Z16 = c16 * sqrt(12) * (10*r^5 - 12*r^3 + 3*r) * cos(a)
        R16 = math.sqrt(12) * (10 * r**5 - 12 * r**3 + 3 * r)
        dR16dr = math.sqrt(12) * (50 * r**4 - 36 * r**2 + 3)
        T16 = cos_a
        dT16dtheta = -sin_a
        dZdx += self.z_coeff[15] * (dR16dr * drdx * T16 + R16 * dT16dtheta * dthetadx)
        dZdy += self.z_coeff[15] * (dR16dr * drdy * T16 + R16 * dT16dtheta * dthetady)

        # Z17 = c17 * sqrt(12) * (10*r^5 - 12*r^3 + 3*r) * sin(a)
        R17 = math.sqrt(12) * (10 * r**5 - 12 * r**3 + 3 * r)
        dR17dr = math.sqrt(12) * (50 * r**4 - 36 * r**2 + 3)
        T17 = sin_a
        dT17dtheta = cos_a
        dZdx += self.z_coeff[16] * (dR17dr * drdx * T17 + R17 * dT17dtheta * dthetadx)
        dZdy += self.z_coeff[16] * (dR17dr * drdy * T17 + R17 * dT17dtheta * dthetady)

        # Z18 = c18 * sqrt(12) * (5*r^5 - 4*r^3) * cos(3*a)
        R18 = math.sqrt(12) * (5 * r**5 - 4 * r**3)
        dR18dr = math.sqrt(12) * (25 * r**4 - 12 * r**2)
        T18 = cos_3a
        dT18dtheta = -3 * sin_3a
        dZdx += self.z_coeff[17] * (dR18dr * drdx * T18 + R18 * dT18dtheta * dthetadx)
        dZdy += self.z_coeff[17] * (dR18dr * drdy * T18 + R18 * dT18dtheta * dthetady)

        # Z19 = c19 * sqrt(12) * (5*r^5 - 4*r^3) * sin(3*a)
        R19 = math.sqrt(12) * (5 * r**5 - 4 * r**3)
        dR19dr = math.sqrt(12) * (25 * r**4 - 12 * r**2)
        T19 = sin_3a
        dT19dtheta = 3 * cos_3a
        dZdx += self.z_coeff[18] * (dR19dr * drdx * T19 + R19 * dT19dtheta * dthetadx)
        dZdy += self.z_coeff[18] * (dR19dr * drdy * T19 + R19 * dT19dtheta * dthetady)

        # Z20 = c20 * sqrt(12) * r^5 * cos(5*a)
        R20 = math.sqrt(12) * r**5
        dR20dr = math.sqrt(12) * 5 * r**4
        T20 = cos_5a
        dT20dtheta = -5 * sin_5a
        dZdx += self.z_coeff[19] * (dR20dr * drdx * T20 + R20 * dT20dtheta * dthetadx)
        dZdy += self.z_coeff[19] * (dR20dr * drdy * T20 + R20 * dT20dtheta * dthetady)

        # Z21 = c21 * sqrt(12) * r^5 * sin(5*a)
        R21 = math.sqrt(12) * r**5
        dR21dr = math.sqrt(12) * 5 * r**4
        T21 = sin_5a
        dT21dtheta = 5 * cos_5a
        dZdx += self.z_coeff[20] * (dR21dr * drdx * T21 + R21 * dT21dtheta * dthetadx)
        dZdy += self.z_coeff[20] * (dR21dr * drdy * T21 + R21 * dT21dtheta * dthetady)

        # Z22 = c22 * sqrt(7) * (20*r^6 - 30*r^4 + 12*r^2 - 1)
        dR22dr = math.sqrt(7) * (120 * r**5 - 120 * r**3 + 24 * r)
        dZdx += self.z_coeff[21] * dR22dr * drdx
        dZdy += self.z_coeff[21] * dR22dr * drdy

        # Z23 = c23 * sqrt(14) * (15*r^6 - 20*r^4 + 6*r^2) * sin(2*a)
        R23 = math.sqrt(14) * (15 * r**6 - 20 * r**4 + 6 * r**2)
        dR23dr = math.sqrt(14) * (90 * r**5 - 80 * r**3 + 12 * r)
        T23 = sin_2a
        dT23dtheta = 2 * cos_2a
        dZdx += self.z_coeff[22] * (dR23dr * drdx * T23 + R23 * dT23dtheta * dthetadx)
        dZdy += self.z_coeff[22] * (dR23dr * drdy * T23 + R23 * dT23dtheta * dthetady)

        # Z24 = c24 * sqrt(14) * (15*r^6 - 20*r^4 + 6*r^2) * cos(2*a)
        R24 = math.sqrt(14) * (15 * r**6 - 20 * r**4 + 6 * r**2)
        dR24dr = math.sqrt(14) * (90 * r**5 - 80 * r**3 + 12 * r)
        T24 = cos_2a
        dT24dtheta = -2 * sin_2a
        dZdx += self.z_coeff[23] * (dR24dr * drdx * T24 + R24 * dT24dtheta * dthetadx)
        dZdy += self.z_coeff[23] * (dR24dr * drdy * T24 + R24 * dT24dtheta * dthetady)

        # Z25 = c25 * sqrt(14) * (6*r^6 - 5*r^4) * sin(4*a)
        R25 = math.sqrt(14) * (6 * r**6 - 5 * r**4)
        dR25dr = math.sqrt(14) * (36 * r**5 - 20 * r**3)
        T25 = sin_4a
        dT25dtheta = 4 * cos_4a
        dZdx += self.z_coeff[24] * (dR25dr * drdx * T25 + R25 * dT25dtheta * dthetadx)
        dZdy += self.z_coeff[24] * (dR25dr * drdy * T25 + R25 * dT25dtheta * dthetady)

        # Z26 = c26 * sqrt(14) * (6*r^6 - 5*r^4) * cos(4*a)
        R26 = math.sqrt(14) * (6 * r**6 - 5 * r**4)
        dR26dr = math.sqrt(14) * (36 * r**5 - 20 * r**3)
        T26 = cos_4a
        dT26dtheta = -4 * sin_4a
        dZdx += self.z_coeff[25] * (dR26dr * drdx * T26 + R26 * dT26dtheta * dthetadx)
        dZdy += self.z_coeff[25] * (dR26dr * drdy * T26 + R26 * dT26dtheta * dthetady)

        # Z27 = c27 * sqrt(14) * r^6 * sin(6*a)
        R27 = math.sqrt(14) * r**6
        dR27dr = math.sqrt(14) * 6 * r**5
        T27 = sin_6a
        dT27dtheta = 6 * cos_6a
        dZdx += self.z_coeff[26] * (dR27dr * drdx * T27 + R27 * dT27dtheta * dthetadx)
        dZdy += self.z_coeff[26] * (dR27dr * drdy * T27 + R27 * dT27dtheta * dthetady)

        # Z28 = c28 * sqrt(14) * r^6 * cos(6*a)
        R28 = math.sqrt(14) * r**6
        dR28dr = math.sqrt(14) * 6 * r**5
        T28 = cos_6a
        dT28dtheta = -6 * sin_6a
        dZdx += self.z_coeff[27] * (dR28dr * drdx * T28 + R28 * dT28dtheta * dthetadx)
        dZdy += self.z_coeff[27] * (dR28dr * drdy * T28 + R28 * dT28dtheta * dthetady)

        # Z29 = c29 * 4 * (35*r^7 - 60*r^5 + 30*r^3 - 4*r) * sin(a)
        R29 = 4 * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r)
        dR29dr = 4 * (245 * r**6 - 300 * r**4 + 90 * r**2 - 4)
        T29 = sin_a
        dT29dtheta = cos_a
        dZdx += self.z_coeff[28] * (dR29dr * drdx * T29 + R29 * dT29dtheta * dthetadx)
        dZdy += self.z_coeff[28] * (dR29dr * drdy * T29 + R29 * dT29dtheta * dthetady)

        # Z30 = c30 * 4 * (35*r^7 - 60*r^5 + 30*r^3 - 4*r) * cos(a)
        R30 = 4 * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r)
        dR30dr = 4 * (245 * r**6 - 300 * r**4 + 90 * r**2 - 4)
        T30 = cos_a
        dT30dtheta = -sin_a
        dZdx += self.z_coeff[29] * (dR30dr * drdx * T30 + R30 * dT30dtheta * dthetadx)
        dZdy += self.z_coeff[29] * (dR30dr * drdy * T30 + R30 * dT30dtheta * dthetady)

        # Z31 = c31 * 4 * (21*r^7 - 30*r^5 + 10*r^3) * sin(3*a)
        R31 = 4 * (21 * r**7 - 30 * r**5 + 10 * r**3)
        dR31dr = 4 * (147 * r**6 - 150 * r**4 + 30 * r**2)
        T31 = sin_3a
        dT31dtheta = 3 * cos_3a
        dZdx += self.z_coeff[30] * (dR31dr * drdx * T31 + R31 * dT31dtheta * dthetadx)
        dZdy += self.z_coeff[30] * (dR31dr * drdy * T31 + R31 * dT31dtheta * dthetady)

        # Z32 = c32 * 4 * (21*r^7 - 30*r^5 + 10*r^3) * cos(3*a)
        R32 = 4 * (21 * r**7 - 30 * r**5 + 10 * r**3)
        dR32dr = 4 * (147 * r**6 - 150 * r**4 + 30 * r**2)
        T32 = cos_3a
        dT32dtheta = -3 * sin_3a
        dZdx += self.z_coeff[31] * (dR32dr * drdx * T32 + R32 * dT32dtheta * dthetadx)
        dZdy += self.z_coeff[31] * (dR32dr * drdy * T32 + R32 * dT32dtheta * dthetady)

        # Z33 = c33 * 4 * (7*r^7 - 6*r^5) * sin(5*a)
        R33 = 4 * (7 * r**7 - 6 * r**5)
        dR33dr = 4 * (49 * r**6 - 30 * r**4)
        T33 = sin_5a
        dT33dtheta = 5 * cos_5a
        dZdx += self.z_coeff[32] * (dR33dr * drdx * T33 + R33 * dT33dtheta * dthetadx)
        dZdy += self.z_coeff[32] * (dR33dr * drdy * T33 + R33 * dT33dtheta * dthetady)

        # Z34 = c34 * 4 * (7*r^7 - 6*r^5) * cos(5*a)
        R34 = 4 * (7 * r**7 - 6 * r**5)
        dR34dr = 4 * (49 * r**6 - 30 * r**4)
        T34 = cos_5a
        dT34dtheta = -5 * sin_5a
        dZdx += self.z_coeff[33] * (dR34dr * drdx * T34 + R34 * dT34dtheta * dthetadx)
        dZdy += self.z_coeff[33] * (dR34dr * drdy * T34 + R34 * dT34dtheta * dthetady)

        # Z35 = c35 * 4 * r^7 * sin(7*a)
        R35 = 4 * r**7
        dR35dr = 4 * 7 * r**6
        T35 = sin_7a
        dT35dtheta = 7 * cos_7a
        dZdx += self.z_coeff[34] * (dR35dr * drdx * T35 + R35 * dT35dtheta * dthetadx)
        dZdy += self.z_coeff[34] * (dR35dr * drdy * T35 + R35 * dT35dtheta * dthetady)

        # Z36 = c36 * 4 * r^7 * cos(7*a)
        R36 = 4 * r**7
        dR36dr = 4 * 7 * r**6
        T36 = cos_7a
        dT36dtheta = -7 * sin_7a
        dZdx += self.z_coeff[35] * (dR36dr * drdx * T36 + R36 * dT36dtheta * dthetadx)
        dZdy += self.z_coeff[35] * (dR36dr * drdy * T36 + R36 * dT36dtheta * dthetady)

        # Z37 = c37 * 3 * (70*r^8 - 140*r^6 + 90*r^4 - 20*r^2 + 1)
        dR37dr = 3 * (560 * r**7 - 840 * r**5 + 360 * r**3 - 40 * r)
        dZdx += self.z_coeff[36] * dR37dr * drdx
        dZdy += self.z_coeff[36] * dR37dr * drdy

        # Apply circular mask
        mask = torch.gt(x_norm**2 + y_norm**2, 1)
        dZdx = torch.where(mask, torch.zeros_like(dZdx), dZdx)
        dZdy = torch.where(mask, torch.zeros_like(dZdy), dZdy)

        # Scale by normalization radius
        dZdx = dZdx / self.norm_radii
        dZdy = dZdy / self.norm_radii

        return dZdx, dZdy

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Generate optimizer parameters for Zernike coefficients."""
        params = []
        self.z_coeff.requires_grad = True
        params.append({"params": [self.z_coeff], "lr": lrs[0]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    # =========================================
    # IO
    # =========================================
    def save_ckpt(self, save_path="./zernike_doe.pth"):
        """Save Zernike DOE coefficients."""
        torch.save(
            {
                "param_model": "zernike",
                "z_coeff": self.z_coeff.clone().detach().cpu(),
                "zernike_order": self.zernike_order,
            },
            save_path,
        )

    def load_ckpt(self, load_path="./zernike_doe.pth"):
        """Load Zernike DOE coefficients."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.z_coeff = ckpt["z_coeff"].to(self.device)
        self.zernike_order = ckpt["zernike_order"]

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "z_coeff": self.z_coeff.clone().detach().cpu().tolist(),
            "zernike_order": self.zernike_order,
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

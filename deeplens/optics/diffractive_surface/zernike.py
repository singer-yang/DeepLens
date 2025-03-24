"""Zernike DOE parameterization."""

import math
import torch
from .base import DiffractiveSurface

class Zernike(DiffractiveSurface):
    """DOE parameterized by Zernike polynomials."""
    
    def __init__(self, d, size, z_coeff=None, zernike_order=37, res=(2000, 2000), mat="fused_silica", fab_ps=0.001, device="cpu"):
        """Initialize Zernike DOE.
        
        Args:
            r: DOE radius
            d: DOE position
            res: DOE resolution
            n_coeffs: Number of Zernike coefficients to use
            fab_ps: Fabrication pixel size
            device: Computation device
        """
        super().__init__(d=d, size=size, res=res, mat=mat, fab_ps=fab_ps, device=device)
        
        # Initialize Zernike coefficients with random values
        assert zernike_order==37, "Currently, Zernike DOE only supports 37 orders"
        self.zernike_order = zernike_order
        if z_coeff is None:
            self.z_coeff = torch.randn(zernike_order, device=self.device) * 1e-3
        else:
            self.z_coeff = z_coeff

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Zernike DOE from a dict."""
        size = doe_dict["size"]
        d = doe_dict["d"]
        res = doe_dict.get("res", (2000, 2000))
        fab_ps = doe_dict.get("fab_ps", 0.001)
        z_coeff = doe_dict.get("z_coeff", None)
        return cls(
            size=size,
            d=d,
            res=res,
            fab_ps=fab_ps,
            z_coeff=z_coeff,
        )

    def _phase_map0(self):
        """Get the phase map at design wavelength."""
        return calculate_zernike_phase(self.z_coeff, grid=self.res[0])

    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self):
        """Activate gradients for optimization."""
        self.z_coeff.requires_grad = True
        
    def get_optimizer_params(self, lr=None):
        """Get parameters for optimization."""
        self.activate_grad()
        lr = 0.01 if lr is None else lr
        return [{"params": [self.z_coeff], "lr": lr}]
    
    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["z_coeff"] = self.z_coeff.clone().detach().cpu()
        surf_dict["zernike_order"] = self.zernike_order
        return surf_dict


def calculate_zernike_phase(z_coeff, grid=256):
    """Calculate phase map produced by Zernike polynomials.
    
    Args:
        z_coeff: Zernike coefficients
        grid: Grid size for phase map
        
    Returns:
        Phase map
    """
    # Generate meshgrid
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, grid), torch.linspace(1, -1, grid), indexing="xy"
    )
    r = torch.sqrt(x**2 + y**2)
    alpha = torch.atan2(y, x)

    # Calculate Zernike polynomials
    Z1 = z_coeff[0] * 1
    Z2 = z_coeff[1] * 2 * r * torch.sin(alpha)
    Z3 = z_coeff[2] * 2 * r * torch.cos(alpha)
    Z4 = z_coeff[3] * math.sqrt(3) * (2 * r**2 - 1)
    Z5 = z_coeff[4] * math.sqrt(6) * r**2 * torch.sin(2 * alpha)
    Z6 = z_coeff[5] * math.sqrt(6) * r**2 * torch.cos(2 * alpha)
    Z7 = z_coeff[6] * math.sqrt(8) * (3 * r**3 - 2 * r) * torch.sin(alpha)
    Z8 = z_coeff[7] * math.sqrt(8) * (3 * r**3 - 2 * r) * torch.cos(alpha)
    Z9 = z_coeff[8] * math.sqrt(8) * r**3 * torch.sin(3 * alpha)
    Z10 = z_coeff[9] * math.sqrt(8) * r**3 * torch.cos(3 * alpha)
    Z11 = z_coeff[10] * math.sqrt(5) * (6 * r**4 - 6 * r**2 + 1)
    Z12 = z_coeff[11] * math.sqrt(10) * (4 * r**4 - 3 * r**2) * torch.cos(2 * alpha)
    Z13 = z_coeff[12] * math.sqrt(10) * (4 * r**4 - 3 * r**2) * torch.sin(2 * alpha)
    Z14 = z_coeff[13] * math.sqrt(10) * r**4 * torch.cos(4 * alpha)
    Z15 = z_coeff[14] * math.sqrt(10) * r**4 * torch.sin(4 * alpha)
    Z16 = (
        z_coeff[15] * math.sqrt(12) * (10 * r**5 - 12 * r**3 + 3 * r) * torch.cos(alpha)
    )
    Z17 = (
        z_coeff[16] * math.sqrt(12) * (10 * r**5 - 12 * r**3 + 3 * r) * torch.sin(alpha)
    )
    Z18 = z_coeff[17] * math.sqrt(12) * (5 * r**5 - 4 * r**3) * torch.cos(3 * alpha)
    Z19 = z_coeff[18] * math.sqrt(12) * (5 * r**5 - 4 * r**3) * torch.sin(3 * alpha)
    Z20 = z_coeff[19] * math.sqrt(12) * r**5 * torch.cos(5 * alpha)
    Z21 = z_coeff[20] * math.sqrt(12) * r**5 * torch.sin(5 * alpha)
    Z22 = z_coeff[21] * math.sqrt(7) * (20 * r**6 - 30 * r**4 + 12 * r**2 - 1)
    Z23 = (
        z_coeff[22]
        * math.sqrt(14)
        * (15 * r**6 - 20 * r**4 + 6 * r**2)
        * torch.sin(2 * alpha)
    )
    Z24 = (
        z_coeff[23]
        * math.sqrt(14)
        * (15 * r**6 - 20 * r**4 + 6 * r**2)
        * torch.cos(2 * alpha)
    )
    Z25 = z_coeff[24] * math.sqrt(14) * (6 * r**6 - 5 * r**4) * torch.sin(4 * alpha)
    Z26 = z_coeff[25] * math.sqrt(14) * (6 * r**6 - 5 * r**4) * torch.cos(4 * alpha)
    Z27 = z_coeff[26] * math.sqrt(14) * r**6 * torch.sin(6 * alpha)
    Z28 = z_coeff[27] * math.sqrt(14) * r**6 * torch.cos(6 * alpha)
    Z29 = z_coeff[28] * 4 * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4) * torch.sin(alpha)
    Z30 = z_coeff[29] * 4 * (35 * r**7 - 60 * r**5 + 30 * r**3 - 4) * torch.cos(alpha)
    Z31 = z_coeff[30] * 4 * (21 * r**7 - 30 * r**5 + 10 * r**3) * torch.sin(3 * alpha)
    Z32 = z_coeff[31] * 4 * (21 * r**7 - 30 * r**5 + 10 * r**3) * torch.cos(3 * alpha)
    Z33 = z_coeff[32] * 4 * (7 * r**7 - 6 * r**5) * torch.sin(5 * alpha)
    Z34 = z_coeff[33] * 4 * (7 * r**7 - 6 * r**5) * torch.cos(5 * alpha)
    Z35 = z_coeff[34] * 4 * r**7 * torch.sin(7 * alpha)
    Z36 = z_coeff[35] * 4 * r**7 * torch.cos(7 * alpha)
    Z37 = z_coeff[36] * 3 * (70 * r**8 - 140 * r**6 + 90 * r**4 - 20 * r**2 + 1)

    # Sum all Zernike terms
    ZW = (
        Z1 + Z2 + Z3 + Z4 + Z5 + Z6 + Z7 + Z8 + Z9 + Z10 +
        Z11 + Z12 + Z13 + Z14 + Z15 + Z16 + Z17 + Z18 + Z19 + Z20 +
        Z21 + Z22 + Z23 + Z24 + Z25 + Z26 + Z27 + Z28 + Z29 + Z30 +
        Z31 + Z32 + Z33 + Z34 + Z35 + Z36 + Z37
    )

    # Apply circular mask
    mask = torch.gt(x**2 + y**2, 1)
    ZW[mask] = 0.0

    return ZW

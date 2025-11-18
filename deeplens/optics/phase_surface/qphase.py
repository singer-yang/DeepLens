"""Q-type surface parameterization for diffractive surfaces."""

import torch

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class QPhase(Phase):
    """Q-type surface profile for diffractive surfaces.

    Q-type surfaces modify the conic constant in the surface equation.
    The surface sag is given by:
    z = (c * r²) / (1 + sqrt(1 - (1+k+Q) * c² * r²))

    Reference:
        [1] https://optics.ansys.com/hc/en-us/articles/42661802686355-Aspheric-Surfaces-Part-1-Introduction-to-Aspherical-Surfaces-in-Optical-Design
        [2] https://community.zemax.com/people-pointers-9/listing-the-q-type-freeform-polynomials-and-surface-equations-5194
    """

    def __init__(
        self,
        r,
        d,
        c=0.01,
        k=0.0,
        Q=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=True,
        device="cpu",
    ):
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

        # Initialize Q-type surface parameters
        self.c = torch.tensor(c)  # curvature
        self.k = torch.tensor(k)  # conic constant
        self.Q = torch.tensor(Q)  # Q-type coefficient

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize Q-type parameters."""
        self.param_model = "qtype"
        self.to(self.device)

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        phi = self._qtype_sag(x, y) * (2 * torch.pi / 0.55e-3)  # Convert to phase (wavelength = 550nm)
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        # Get surface derivatives
        dz_dx, dz_dy = self._dfdxy(x, y)

        # Convert to phase derivatives (multiply by 2π/λ)
        wavelength = 0.55e-3  # 550nm design wavelength
        dphidx = dz_dx * (2 * torch.pi / wavelength)
        dphidy = dz_dy * (2 * torch.pi / wavelength)

        return dphidx, dphidy

    def _qtype_sag(self, x, y):
        """Compute Q-type surface sag."""
        r2 = x**2 + y**2
        # Q-type surface equation: z = (c * r²) / (1 + sqrt(1 - (1+k+Q) * c² * r²))
        denominator = 1 + torch.sqrt(1 - (1 + self.k + self.Q) * self.c**2 * r2 + EPSILON)
        sag = self.c * r2 / denominator
        return sag

    def _dfdxy(self, x, y):
        """Compute first-order surface derivatives."""
        r2 = x**2 + y**2

        # Avoid division by zero
        sqrt_term = torch.sqrt(1 - (1 + self.k + self.Q) * self.c**2 * r2 + EPSILON)
        denominator = 1 + sqrt_term

        # Derivative of sag with respect to r²
        d_sag_dr2 = self.c / denominator - self.c * r2 * (-0.5) * (1 + self.k + self.Q) * self.c**2 / (denominator * sqrt_term)

        # Convert to derivatives with respect to x and y
        dz_dx = d_sag_dr2 * 2 * x
        dz_dy = d_sag_dr2 * 2 * y

        return dz_dx, dz_dy

    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters."""
        params = []

        # Optimize position
        self.d.requires_grad = True
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize Q-type surface parameters
        self.c.requires_grad = True
        params.append({"params": [self.c], "lr": lrs[1]})

        self.k.requires_grad = True
        params.append({"params": [self.k], "lr": lrs[2]})

        self.Q.requires_grad = True
        params.append({"params": [self.Q], "lr": lrs[3]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./qtype_doe.pth"):
        """Save Q-type DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "c": self.c.clone().detach().cpu(),
                "k": self.k.clone().detach().cpu(),
                "Q": self.Q.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./qtype_doe.pth"):
        """Load Q-type DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.c = ckpt["c"].to(self.device)
        self.k = ckpt["k"].to(self.device)
        self.Q = ckpt["Q"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "param_model": self.param_model,
            "c": round(self.c.item(), 6),
            "k": round(self.k.item(), 6),
            "Q": round(self.Q.item(), 6),
            "norm_radii": round(self.norm_radii, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

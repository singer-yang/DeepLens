"""Grating phase on a plane surface."""

import torch

from deeplens.optics.phase_surface.phase import Phase


class GratingPhase(Phase):
    """Grating phase on a plane surface."""

    def __init__(
        self,
        r,
        d,
        theta=0.0,
        alpha=0.0,
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

        # Grating parameters
        self.theta = torch.tensor(theta)  # angle from x-axis to grating vector
        self.alpha = torch.tensor(alpha)  # slope of the grating

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize grating parameters."""
        self.param_model = "grating"
        self.to(self.device)

    def phi(self, x, y):
        """Reference phase map at design wavelength."""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        dphidx = self.alpha * torch.sin(self.theta) / self.norm_radii
        dphidy = self.alpha * torch.cos(self.theta) / self.norm_radii
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-3], optim_mat=False):
        """Generate optimizer parameters."""
        params = []

        # Optimize grating parameters
        self.theta.requires_grad = True
        self.alpha.requires_grad = True
        params.append({"params": [self.theta], "lr": lrs[0]})
        params.append({"params": [self.alpha], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """Save grating DOE parameters."""
        torch.save(
            {
                "param_model": self.param_model,
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """Load grating DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "param_model": self.param_model,
            "theta": round(self.theta.item(), 4),
            "alpha": round(self.alpha.item(), 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

"""Base phase surface class for diffractive surfaces (metasurface or DOE)."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deeplens.basics import EPSILON
from deeplens.optics.geometric_surface.plane import Plane


class Phase(Plane):
    """Base phase profile for diffractive surfaces (metasurface or DOE).

    This is the base class that provides common functionality for all phase parameterizations.
    Specific parameterizations should inherit from this class.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
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
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # DOE geometry
        self.r = r
        self.w = r * float(np.sqrt(2))
        self.h = r * float(np.sqrt(2))

        # Use ray tracing to simulate diffraction, the same as Zemax
        self.diffraction = True
        self.diffraction_order = 1
        self.norm_radii = self.r if norm_radii is None else norm_radii

        self.to(device)

    # ==============================
    # Abstract methods to be implemented by subclasses
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength. Must be implemented by subclasses."""
        raise NotImplementedError("phi() must be implemented by subclasses")

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives. Must be implemented by subclasses."""
        raise NotImplementedError("dphi_dxy() must be implemented by subclasses")

    def init_param_model(self):
        """Initialize parameterization parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "init_param_model() must be implemented by subclasses"
        )

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "get_optimizer_params() must be implemented by subclasses"
        )

    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("save_ckpt() must be implemented by subclasses")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("load_ckpt() must be implemented by subclasses")

    def surf_dict(self):
        """Return surface parameters. Must be implemented by subclasses."""
        raise NotImplementedError("surf_dict() must be implemented by subclasses")

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def activate_diffraction(self, diffraction_order=1):
        """Activate diffraction of DOE in ray tracing."""
        self.diffraction = True
        self.diffraction_order = diffraction_order
        print("Diffraction of DOE in ray tracing is enabled.")

    def ray_reaction(self, ray, n1=None, n2=None):
        """Ray reaction on DOE surface."""
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.refract(ray, n1 / n2)
        if self.diffraction:
            ray = self.diffract(ray)
        ray = self.to_global_coord(ray)
        return ray

    def diffract(self, ray):
        """Diffraction of DOE surface.
            1, The phase φ in radians adds to the optical path length of the ray
            2, The gradient of the phase profile (phase slope) change the direction of rays.

        Reference:
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 2011.
        """
        forward = (ray.d * ray.valid.unsqueeze(-1))[..., 2].sum() > 0
        valid = ray.valid > 0

        # Diffraction 1: DOE phase modulation
        if ray.coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1])
            new_opl = ray.opl + phi.unsqueeze(-1) * (ray.wvln * 1e-3) / (2 * torch.pi)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        # Diffraction 2: bend rays
        # Perpendicular incident rays are diffracted following (1) grating equation and (2) local grating approximation
        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

        wvln_mm = ray.wvln.squeeze(-1) * 1e-3
        order = self.diffraction_order
        if forward:
            new_d_x = ray.d[..., 0] + wvln_mm / (2 * torch.pi) * dphidx * order
            new_d_y = ray.d[..., 1] + wvln_mm / (2 * torch.pi) * dphidy * order
        else:
            new_d_x = ray.d[..., 0] - wvln_mm / (2 * torch.pi) * dphidx * order
            new_d_y = ray.d[..., 1] - wvln_mm / (2 * torch.pi) * dphidy * order

        new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        return ray

    def _sag(self, x, y):
        """Diffractive surface is now attached to a plane surface."""
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """Diffractive surface is now attached to a plane surface."""
        return torch.zeros_like(x), torch.zeros_like(y)

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points.

        Normal vector points from the surface toward the side where the light is coming from.
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1
        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def surface_with_offset(self, *args, **kwargs):
        """Surface sag with offset, only used in layout drawing."""
        return self.d

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer(self, lrs):
        """Generate optimizer.

        Args:
            lrs (list or float): Learning rates for different parameters.
        """
        if isinstance(lrs, float):
            lrs = [lrs]
        assert self.diffraction, "Diffraction is not activated yet."
        params = self.get_optimizer_params(lrs)
        optimizer = torch.optim.Adam(params)
        return optimizer

    # =========================================
    # Visualization
    # =========================================
    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw height map. Range from [0, max_height]."""
        x, y = torch.meshgrid(
            torch.linspace(-self.l / 2, self.l / 2, 2000),
            torch.linspace(self.l / 2, -self.l / 2, 2000),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * torch.pi)
        ax[0].set_title("Phase map 0.55um", fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="black", linestyle="-"):
        """Draw DOE as a two-side surface."""
        max_offset = self.d.item() / 100
        d = self.d.item()

        # Draw DOE
        roc = self.r * 2
        x = np.linspace(-self.r, self.r, 128)
        y = np.zeros_like(x)
        r = np.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - np.sqrt(1 - r**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color="orange", linestyle=linestyle, linewidth=0.75)

        # # Draw DOE base
        # z_bound = [
        #     d + sag[0],
        #     d + sag[0] + max_offset,
        #     d + sag[0] + max_offset,
        #     d + sag[-1],
        # ]
        # x_bound = [-self.r, -self.r, self.r, self.r]
        # ax.plot(z_bound, x_bound, color=color, linestyle=linestyle, linewidth=0.75)

    # =========================================
    # IO
    # =========================================

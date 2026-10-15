"""Thin lens element. Both sides are air."""

import torch
import torch.nn.functional as F

from .plane import Plane


class ThinLens(Plane):
    """Ideal thin lens with air on both sides.

    A zero-thickness paraxial lens of focal length `f` [mm] in a sequential
    optical train. It refracts every ray toward the on-axis focal point
    (back focus for `f` greater than 0, front virtual focus for `f` less than
    0) with no aberrations, and in coherent mode applies the ideal quadratic
    phase (optical path length) of a Fresnel lens. Inherits the planar
    geometry from `Plane`.

    Attributes:
        f (torch.Tensor): Focal length [mm], scalar tensor.
        d_next (torch.Tensor): Axial thickness to the next vertex [mm].
        r (float): Aperture radius [mm] (half-diagonal if `is_square`).
    """

    def __init__(
        self,
        r,
        d_next,
        f=100.0,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize a thin lens surface.

        Args:
            r (float): Aperture radius [mm] (half-diagonal if `is_square`).
            d_next (float): Axial thickness to the next vertex [mm].
            f (float, optional): Focal length [mm]; positive converges,
                negative diverges. Defaults to 100.0.
            pos_xy (list[float], optional): Lateral offset [x, y] [mm].
                Defaults to [0.0, 0.0].
            vec_local (list[float], optional): Local surface normal direction;
                normalized internally. Defaults to [0.0, 0.0, 1.0] (on-axis).
            is_square (bool, optional): Use a square aperture instead of a
                circular one. Defaults to False.
            device (str, optional): Compute device. Defaults to "cpu".
        """
        Plane.__init__(
            self,
            r=r,
            d_next=d_next,
            mat2="air",
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.f = torch.as_tensor(f, device=device, dtype=self.d_next.dtype)

    def set_f(self, f):
        """Set the focal length.

        Args:
            f (float): New focal length [mm].
        """
        self.f = torch.as_tensor(f, device=self.device, dtype=self.dtype)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a ThinLens from a surface dictionary.

        Args:
            surf_dict (dict): Surface parameters with keys "r", "d_next", and "f".

        Returns:
            thinlens (ThinLens): The constructed thin lens surface.
        """
        return cls(
            surf_dict["r"],
            surf_dict["d_next"],
            surf_dict["f"],
            pos_xy=surf_dict.get("pos_xy", [0.0, 0.0]),
            vec_local=surf_dict.get("vec_local", [0.0, 0.0, 1.0]),
            is_square=surf_dict.get("is_square", False),
            device=surf_dict.get("device", "cpu"),
        )

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """Activate gradients on the axial position and focal length and return optimizer parameter groups.

        Args:
            lrs (list[float], optional): Learning rates; `lrs[0]` is applied to
                `d_next` and `lrs[1]` to the focal length `f`.
                Defaults to [1e-4, 1e-4].
            optim_mat (bool, optional): Unused for a thin lens (both sides are
                air). Defaults to False.

        Returns:
            params (list[dict]): Optimizer parameter groups, each a dict with
                "params" and "lr" keys.
        """
        params = []

        self.d_next.requires_grad_(True)
        params.append({"params": [self.d_next], "lr": lrs[0]})

        self.f.requires_grad_(True)
        params.append({"params": [self.f], "lr": lrs[1]})

        return params

    def refract(self, ray, eta=1.0):
        """Refract a ray bundle through the ideal thin lens.

        Rays sharing a direction converge to a common point on the focal plane.
        The convergence point is found by parallel-shifting each ray's direction
        to the lens center; the new direction points from the ray's intersection
        toward that point (away from it for a diverging lens). In coherent mode
        the optical path length receives the ideal quadratic phase of a Fresnel
        lens, $\\Delta\\,\\text{opl} = \\mp (x^2 + y^2) / (2 f d_z)$ for
        forward/backward propagation.

        Args:
            ray (Ray): Incident ray bundle in the local frame. Origins `ray.o`
                and directions `ray.d` have shape (..., num_rays, 3) [mm].
            eta (float, optional): Refractive index ratio; unused for an ideal
                thin lens. Defaults to 1.0.

        Returns:
            ray (Ray): The same bundle with `d` (and, in coherent mode, `opl`)
                updated in place.
        """
        forward = (ray.d * ray.is_valid.unsqueeze(-1))[..., 2].sum() > 0

        # Calculate convergence point
        if forward:
            t0 = self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = self.f.view(1).expand_as(xy_final[..., 0].unsqueeze(-1))
            o_final = torch.cat([xy_final, z_final], dim=-1)
        else:
            t0 = -self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = (-self.f).view(1).expand_as(xy_final[..., 0].unsqueeze(-1))
            o_final = torch.cat([xy_final, z_final], dim=-1)

        # New ray direction
        if self.f > 0:
            new_d = o_final - ray.o
        else:
            new_d = ray.o - o_final
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = new_d

        # Optical path length change
        if ray.is_coherent:
            valid = ray.is_valid > 0
            if forward:
                new_opl = (
                    ray.opl
                    - (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
                ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)
            else:
                new_opl = (
                    ray.opl
                    + (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
                ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def paraxial_power(self, n1, n2, wvln=None):
        """Return the paraxial optical power of this thin lens [1/mm].

        A thin lens is defined by its focal length, so its power is `1 / f`
        regardless of the surrounding media.

        Args:
            n1 (torch.Tensor): Refractive index of the incident medium.
            n2 (torch.Tensor): Refractive index of the transmission medium.
            wvln (float, optional): Wavelength [µm]. Unused by refractive
                surfaces, which see it only through `n1`/`n2`.

        Returns:
            power (torch.Tensor): Surface power `1 / f` [1/mm], scalar.
        """
        return 1.0 / self.f

    def _sag(self, x, y):
        """Return the surface sag, which is identically zero for a thin lens.

        Args:
            x (torch.Tensor): Local x coordinates [mm].
            y (torch.Tensor): Local y coordinates [mm].

        Returns:
            sag (torch.Tensor): Zero sag [mm], same shape as `x`.
        """
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """Return the sag gradient, which is zero in both directions for a thin lens.

        Args:
            x (torch.Tensor): Local x coordinates [mm].
            y (torch.Tensor): Local y coordinates [mm].

        Returns:
            dfdx (torch.Tensor): Partial derivative df/dx [unitless], same shape as `x`.
            dfdy (torch.Tensor): Partial derivative df/dy [unitless], same shape as `x`.
        """
        return torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # Visualization
    # =========================================
    def draw_widget(self, ax, color="black", linestyle="-", d=0.0):
        """Draw the thin lens on a Matplotlib axis.

        Renders a vertical line spanning the aperture with a double-headed
        arrow ("<->" for a converging lens, "]-[" for a diverging lens).

        Args:
            ax (matplotlib.axes.Axes): Target axis for the cross-section plot.
            color (str, optional): Line and arrow color. Defaults to "black".
            linestyle (str, optional): Matplotlib line style. Defaults to "-".
        """
        d = float(d)
        r = self.r

        # Draw a vertical line to represent the thin lens
        ax.plot([d, d], [-r, r], color=color, linestyle=linestyle, linewidth=0.75)

        # Draw arrow to indicate the focal length
        arrowstyle = "<->" if self.f > 0 else "]-["
        ax.annotate(
            "",
            xy=(d, r),
            xytext=(d, -r),
            arrowprops=dict(
                arrowstyle=arrowstyle, color=color, linestyle=linestyle, linewidth=0.75
            ),
        )

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Serialize the surface to a parameter dictionary.

        Returns:
            surf_dict (dict): Surface parameters with keys "type", "f", "r",
                "d_next", and "mat2".
        """
        surf_dict = {
            "type": "ThinLens",
            "f": self.f.item(),
            "r": self.r,
            "d_next": self.d_next.item(),
            "mat2": "air",
            "pos_xy": [self.pos_x.item(), self.pos_y.item()],
            "vec_local": self.vec_local.tolist(),
            "is_square": self.is_square,
        }

        return surf_dict

"""Binary2 phase on a plane substrate."""

import torch

from ..config import EPSILON
from .base_phase import Phase


class Binary2Phase(Phase):
    """Zemax BINARY_2 phase profile on a flat substrate.

    Parameterizes the diffractive phase as an even radial polynomial in the
    normalized radius $\\rho = r / r_\\text{norm}$:

    $$\\phi(\\rho) = \\sum_{i=1}^{6} a_{2i}\\,\\rho^{2i}$$

    where the coefficients `order2`..`order12` are stored in radians [rad].
    The phase is evaluated with Horner's method and wrapped to $[0, 2\\pi)$.

    Attributes:
        order2 (torch.Tensor): Coefficient of $\\rho^2$, scalar [rad].
        order4 (torch.Tensor): Coefficient of $\\rho^4$, scalar [rad].
        order6 (torch.Tensor): Coefficient of $\\rho^6$, scalar [rad].
        order8 (torch.Tensor): Coefficient of $\\rho^8$, scalar [rad].
        order10 (torch.Tensor): Coefficient of $\\rho^{10}$, scalar [rad].
        order12 (torch.Tensor): Coefficient of $\\rho^{12}$, scalar [rad].
        param_model (str): Parameterization tag, always "binary2".
        norm_radii (float): Normalization radius $r_\\text{norm}$ [mm].
    """

    def __init__(
        self,
        r,
        d_next,
        order2=0.0,
        order4=0.0,
        order6=0.0,
        order8=0.0,
        order10=0.0,
        order12=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a Binary2 phase surface.

        Args:
            r (float): Aperture radius (half-diameter) [mm].
            d (float): Axial position of the surface in global coordinates [mm].
            order2 (float, optional): Coefficient of $\\rho^2$ [rad]. Defaults to 0.0.
            order4 (float, optional): Coefficient of $\\rho^4$ [rad]. Defaults to 0.0.
            order6 (float, optional): Coefficient of $\\rho^6$ [rad]. Defaults to 0.0.
            order8 (float, optional): Coefficient of $\\rho^8$ [rad]. Defaults to 0.0.
            order10 (float, optional): Coefficient of $\\rho^{10}$ [rad]. Defaults to 0.0.
            order12 (float, optional): Coefficient of $\\rho^{12}$ [rad]. Defaults to 0.0.
            norm_radii (float or None, optional): Normalization radius for the polynomial [mm].
                Defaults to None, in which case `r` is used.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) offset of the surface center [mm]. Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True, use a square aperture; otherwise circular. Defaults to True.
            device (str, optional): Torch device. Defaults to "cpu".
        """
        super().__init__(
            r=r,
            d_next=d_next,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # Initialize polynomial coefficients
        self.order2 = torch.tensor(order2)
        self.order4 = torch.tensor(order4)
        self.order6 = torch.tensor(order6)
        self.order8 = torch.tensor(order8)
        self.order10 = torch.tensor(order10)
        self.order12 = torch.tensor(order12)

        self.param_model = "binary2"
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a Binary2 phase surface from a parameter dictionary.

        Args:
            surf_dict (dict): Surface parameters. Requires keys "r" and "d_next";
                optionally "order2".."order12", "norm_radii", "mat2", and
                "is_square".

        Returns:
            obj (Binary2Phase): The constructed phase surface.
        """
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        is_square = surf_dict.get("is_square", True)
        obj = cls(
            surf_dict["r"],
            surf_dict["d_next"],
            surf_dict.get("order2", 0.0),
            surf_dict.get("order4", 0.0),
            surf_dict.get("order6", 0.0),
            surf_dict.get("order8", 0.0),
            surf_dict.get("order10", 0.0),
            surf_dict.get("order12", 0.0),
            norm_radii,
            mat2,
            is_square=is_square,
        )
        return obj

    def phi(self, x, y):
        """Compute the reference phase at the design wavelength.

        Evaluates the even radial polynomial in normalized radius via Horner's
        method and wraps the result to $[0, 2\\pi)$ with `torch.remainder`.

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], same shape as `x`.

        Returns:
            phi (torch.Tensor): Phase values [rad] wrapped to $[0, 2\\pi)$, same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # Horner's method: r2*(o2 + r2*(o4 + r2*(o6 + r2*(o8 + r2*(o10 + r2*o12)))))
        phi = r2 * (
            self.order2
            + r2
            * (
                self.order4
                + r2
                * (
                    self.order6
                    + r2 * (self.order8 + r2 * (self.order10 + r2 * self.order12))
                )
            )
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the lateral phase gradient at the given points.

        Differentiates the (unwrapped) phase polynomial and applies the chain
        rule through the normalized radius.

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], same shape as `x`.

        Returns:
            dphidx (torch.Tensor): Partial derivative $\\partial\\phi/\\partial x$ [rad/mm], same shape as `x`.
            dphidy (torch.Tensor): Partial derivative $\\partial\\phi/\\partial y$ [rad/mm], same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # d/dr2 of polynomial, then chain rule: dphi/dx = dphi/dr2 * 2*x_norm / norm_radii
        # Horner's: o2 + r2*(2*o4 + r2*(3*o6 + r2*(4*o8 + r2*(5*o10 + r2*6*o12))))
        dphidr2 = self.order2 + r2 * (
            2 * self.order4
            + r2
            * (
                3 * self.order6
                + r2
                * (4 * self.order8 + r2 * (5 * self.order10 + r2 * 6 * self.order12))
            )
        )
        dphidx = dphidr2 * 2 * x_norm / self.norm_radii
        dphidy = dphidr2 * 2 * y_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Build optimizer parameter groups for the phase surface.

        Enables gradients on `d_next` and the six polynomial coefficients,
        grouping `d_next` with the first learning rate and all
        coefficients with the second.

        Args:
            lrs (list, optional): Learning rates ``[lr_position, lr_coeffs]``. Defaults to [1e-4, 1e-2].
            optim_mat (bool, optional): Must be False; materials are not optimized for phase surfaces. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts for a torch optimizer.

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize sequential thickness.
        self.d_next.requires_grad = True
        params.append({"params": [self.d_next], "lr": lrs[0]})

        # Optimize polynomial coefficients
        self.order2.requires_grad = True
        self.order4.requires_grad = True
        self.order6.requires_grad = True
        self.order8.requires_grad = True
        self.order10.requires_grad = True
        self.order12.requires_grad = True
        params.append({"params": [self.order2], "lr": lrs[1]})
        params.append({"params": [self.order4], "lr": lrs[1]})
        params.append({"params": [self.order6], "lr": lrs[1]})
        params.append({"params": [self.order8], "lr": lrs[1]})
        params.append({"params": [self.order10], "lr": lrs[1]})
        params.append({"params": [self.order12], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./binary2_doe.pth"):
        """Save the Binary2 phase coefficients to disk.

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to "./binary2_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order8": self.order8.clone().detach().cpu(),
                "order10": self.order10.clone().detach().cpu(),
                "order12": self.order12.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./binary2_doe.pth"):
        """Load Binary2 phase coefficients from disk onto the surface device.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to "./binary2_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order8 = ckpt["order8"].to(self.device)
        self.order10 = ckpt["order10"].to(self.device)
        self.order12 = ckpt["order12"].to(self.device)

    def zmx_str(self, surf_idx, d_next):
        """Return the Zemax BINARY_2 surface block as a string.

        PARM 1-8 are set to zero (flat substrate, no aspheric sag) so that
        Zemax interprets the XDAT entries purely as phase polynomial
        coefficients.

        Args:
            surf_idx (int): Surface index used in the SURF header.
            d_next (torch.Tensor): Distance to the next surface [mm], scalar tensor (read via `.item()`).

        Returns:
            zmx_str (str): Multi-line Zemax surface description.
        """
        coeffs = [
            self.order2.item(),
            self.order4.item(),
            self.order6.item(),
            self.order8.item(),
            self.order10.item(),
            self.order12.item(),
        ]
        n_terms = len(coeffs)

        # Build XDAT block: term count, norm radius, then coefficients
        xdat_str = f"    XDAT 1 {n_terms} 0 0\n"
        xdat_str += f"    XDAT 2 {self.norm_radii} 0 0\n"
        for j, coeff in enumerate(coeffs, start=3):
            xdat_str += f"    XDAT {j} {coeff} 0 0\n"

        zmx_str = f"""SURF {surf_idx}
    TYPE BINARY_2
    CURV 0.0
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    PARM 1 0
    PARM 2 0
    PARM 3 0
    PARM 4 0
    PARM 5 0
    PARM 6 0
    PARM 7 0
    PARM 8 0
{xdat_str}"""
        return zmx_str

    def surf_dict(self):
        """Return a serializable dictionary of the surface parameters.

        Returns:
            surf_dict (dict): Surface parameters including type, radius `r` [mm],
                polynomial coefficients, `norm_radii` [mm], position `d` [mm],
                and material name.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": self.order2.item(),
            "order4": self.order4.item(),
            "order6": self.order6.item(),
            "order8": self.order8.item(),
            "order10": self.order10.item(),
            "order12": self.order12.item(),
            "norm_radii": float(self.norm_radii),
            "d_next": round(self.d_next.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict

"""Diffractive surfaces (Local gratings) simulated with ray tracing.

Reference:
    https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .base import EPSILON, Surface


class Diffractive_GEO(Surface):
    def __init__(
        self, r, d, glass="test", param_model="binary2", thickness=0.5, device="cpu"
    ):
        Surface.__init__(self, r, d, mat2="air", is_square=True, device=device)

        # DOE geometry
        self.r = r
        self.w = r * float(np.sqrt(2))
        self.h = r * float(np.sqrt(2))
        self.thickness = thickness
        self.glass = glass

        # Use ray tracing to simulate diffraction, the same as Zemax
        self.diffraction = False
        self.diffraction_order = 1
        print(
            "A Diffractive_GEO is created, but ray-tracing diffraction is not activated."
        )

        self.to(device)
        self.init_param_model(param_model)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "glass" in surf_dict:
            glass = surf_dict["glass"]
        else:
            glass = "test"
        if "thickness" in surf_dict:
            thickness = surf_dict["thickness"]
        else:
            thickness = 0.5
        return cls(
            surf_dict["l"], surf_dict["d"], glass, surf_dict["param_model"], thickness
        )

    def init_param_model(self, param_model="binary2"):
        self.param_model = param_model
        if self.param_model == "fresnel":
            # Focal length at 550nm
            self.f0 = torch.tensor(100.0)

        elif self.param_model == "binary2":
            # Zemax binary2 surface type
            self.order2 = torch.tensor(0.0)
            self.order4 = torch.tensor(0.0)
            self.order6 = torch.tensor(0.0)
            self.order8 = torch.tensor(0.0)

        elif self.param_model == "poly1d":
            rand_value = np.random.rand(6) * 0.001
            self.order2 = torch.tensor(rand_value[0])
            self.order3 = torch.tensor(rand_value[1])
            self.order4 = torch.tensor(rand_value[2])
            self.order5 = torch.tensor(rand_value[3])
            self.order6 = torch.tensor(rand_value[4])
            self.order7 = torch.tensor(rand_value[5])

        elif self.param_model == "grating":
            # A grating surface
            self.theta = torch.tensor(0.0)  # angle from x-axis to grating vector
            self.alpha = torch.tensor(0.0)  # slope of the grating

        else:
            raise ValueError(f"Unsupported parameter model: {self.param_model}")

        self.to(self.device)

    def activate_diffraction(self, diffraction_order=1):
        self.diffraction = True
        self.diffraction_order = diffraction_order
        print("Diffraction of DOE in ray tracing is enabled.")

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def ray_reaction(self, ray, n1=None, n2=None):
        """Ray reaction on DOE surface. Imagine the DOE as a wrapped positive convex lens for debugging.

        1, The phase φ in radians adds to the optical path length of the ray
        2, The gradient of the phase profile (phase slope) change the direction of rays.

        https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0

        # Intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        # valid = (new_o[...,0].abs() <= self.h/2) & (new_o[...,1].abs() <= self.w/2) & (ray.ra > 0) # square
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) <= self.r) & (
            ray.ra > 0
        )  # circular
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            # OPL change
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        if self.diffraction:
            # Diffraction 1: DOE phase modulation
            if ray.coherent:
                phi = self.phi(ray.o[..., 0], ray.o[..., 1])
                new_opl = ray.opl + phi * (ray.wvln * 1e-3) / (2 * np.pi)
                new_opl[~valid] = ray.opl[~valid]
                ray.opl = new_opl

            # Diffraction 2: bend rays
            # Perpendicular incident rays are diffracted following (1) grating equation and (2) local grating approximation
            dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

            if forward:
                new_d_x = (
                    ray.d[..., 0]
                    + (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                )
                new_d_y = (
                    ray.d[..., 1]
                    + (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order
                )
            else:
                new_d_x = (
                    ray.d[..., 0]
                    - (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                )
                new_d_y = (
                    ray.d[..., 1]
                    - (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order
                )

            new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
            new_d = F.normalize(new_d, p=2, dim=-1)

            new_d[~valid] = ray.d[~valid]
            ray.d = new_d

        return ray

    def phi(self, x, y):
        """Reference phase map at design wavelength (independent to wavelength). We have the same definition of phase (phi) as Zemax."""
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        if self.param_model == "fresnel":
            phi = (
                -2 * torch.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1)
            )  # unit [mm]

        elif self.param_model == "binary2":
            phi = (
                self.order2 * r**2
                + self.order4 * r**4
                + self.order6 * r**6
                + self.order8 * r**8
            )

        elif self.param_model == "poly1d":
            phi_even = self.order2 * r**2 + self.order4 * r**4 + self.order6 * r**6
            phi_odd = (
                self.order3 * (x_norm**3 + y_norm**3)
                + self.order5 * (x_norm**5 + y_norm**5)
                + self.order7 * (x_norm**7 + y_norm**7)
            )
            phi = phi_even + phi_odd

        elif self.param_model == "grating":
            phi = self.alpha * (
                x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
            )

        else:
            raise NotImplementedError(
                f"phi() is not implemented for {self.param_model}"
            )

        phi = torch.remainder(phi, 2 * np.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        if self.param_model == "fresnel":
            dphidx = -2 * np.pi * x / (0.55e-3 * self.f0)  # unit [mm]
            dphidy = -2 * np.pi * y / (0.55e-3 * self.f0)

        elif self.param_model == "binary2":
            dphidr = (
                2 * self.order2 * r
                + 4 * self.order4 * r**3
                + 6 * self.order6 * r**5
                + 8 * self.order8 * r**7
            )
            dphidx = dphidr * x_norm / r / self.r
            dphidy = dphidr * y_norm / r / self.r

        elif self.param_model == "poly1d":
            dphi_even_dr = (
                2 * self.order2 * r + 4 * self.order4 * r**3 + 6 * self.order6 * r**5
            )
            dphi_even_dz = dphi_even_dr * x_norm / r / self.r
            dphi_even_dy = dphi_even_dr * y_norm / r / self.r

            dphi_odd_dx = (
                3 * self.order3 * x_norm**2
                + 5 * self.order5 * x_norm**4
                + 7 * self.order7 * x_norm**6
            ) / self.r
            dphi_odd_dy = (
                3 * self.order3 * y_norm**2
                + 5 * self.order5 * y_norm**4
                + 7 * self.order7 * y_norm**6
            ) / self.r

            dphidx = dphi_even_dz + dphi_odd_dx
            dphidy = dphi_even_dy + dphi_odd_dy

        elif self.param_model == "grating":
            dphidx = self.alpha * torch.sin(self.theta) / self.r
            dphidy = self.alpha * torch.cos(self.theta) / self.r

        else:
            raise NotImplementedError(
                f"dphi_dxy() is not implemented for {self.param_model}"
            )

        return dphidx, dphidy

    def _sag(self, x, y):
        raise ValueError(
            "sag() function is meaningless for phase DOE, use phi() function."
        )

    def _dfdxy(self, x, y):
        raise ValueError(
            "_dfdxy() function is meaningless for DOE, use dphi_dxy() function."
        )

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer_params(self, lr=None):
        """Generate optimizer parameters."""
        params = []
        if self.param_model == "binary2":
            lr = 0.001 if lr is None else lr
            self.order2.requires_grad = True
            self.order4.requires_grad = True
            self.order6.requires_grad = True
            self.order8.requires_grad = True
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order8], "lr": lr})

        elif self.param_model == "poly1d":
            lr = 0.001 if lr is None else lr
            self.order2.requires_grad = True
            self.order3.requires_grad = True
            self.order4.requires_grad = True
            self.order5.requires_grad = True
            self.order6.requires_grad = True
            self.order7.requires_grad = True
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order3], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order5], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order7], "lr": lr})

        elif self.param_model == "grating":
            lr = 0.1 if lr is None else lr
            self.theta.requires_grad = True
            self.alpha.requires_grad = True
            params.append({"params": [self.theta], "lr": lr})
            params.append({"params": [self.alpha], "lr": lr})

        else:
            raise NotImplementedError(
                f"get_optimizer_params() is not implemented for {self.param_model}"
            )

        return params

    def get_optimizer(self, lr=None):
        """Generate optimizer.

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-3.
            iterations (float, optional): Iterations. Defaults to 1e4.
        """
        assert self.diffraction, "Diffraction is not activated yet."
        params = self.get_optimizer_params(lr)
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
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * np.pi)
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
        ax.plot(d + sag, x, color=color, linestyle=linestyle, linewidth=0.75)

        # Draw DOE base
        z_bound = [
            d + sag[0],
            d + sag[0] + max_offset,
            d + sag[0] + max_offset,
            d + sag[-1],
        ]
        x_bound = [-self.r, -self.r, self.r, self.r]
        ax.plot(z_bound, x_bound, color=color, linestyle=linestyle, linewidth=0.75)

    # =========================================
    # IO
    # =========================================
    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE height map."""
        if self.param_model == "binary2":
            torch.save(
                {
                    "param_model": self.param_model,
                    "order2": self.order2.clone().detach().cpu(),
                    "order4": self.order4.clone().detach().cpu(),
                    "order6": self.order6.clone().detach().cpu(),
                    "order8": self.order8.clone().detach().cpu(),
                },
                save_path,
            )
        elif self.param_model == "poly1d":
            torch.save(
                {
                    "param_model": self.param_model,
                    "order2": self.order2.clone().detach().cpu(),
                    "order3": self.order3.clone().detach().cpu(),
                    "order4": self.order4.clone().detach().cpu(),
                    "order5": self.order5.clone().detach().cpu(),
                    "order6": self.order6.clone().detach().cpu(),
                    "order7": self.order7.clone().detach().cpu(),
                },
                save_path,
            )
        elif self.param_model == "grating":
            torch.save(
                {
                    "param_model": self.param_model,
                    "theta": self.theta.clone().detach().cpu(),
                    "alpha": self.alpha.clone().detach().cpu(),
                },
                save_path,
            )
        else:
            raise ValueError(f"Unknown parameterization: {self.param_model}")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE height map."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        if self.param_model == "binary2" or self.param_model == "poly_even":
            self.param_model = "binary2"
            self.order2 = ckpt["order2"].to(self.device)
            self.order4 = ckpt["order4"].to(self.device)
            self.order6 = ckpt["order6"].to(self.device)
            self.order8 = ckpt["order8"].to(self.device)
        elif self.param_model == "poly1d":
            self.order2 = ckpt["order2"].to(self.device)
            self.order3 = ckpt["order3"].to(self.device)
            self.order4 = ckpt["order4"].to(self.device)
            self.order5 = ckpt["order5"].to(self.device)
            self.order6 = ckpt["order6"].to(self.device)
            self.order7 = ckpt["order7"].to(self.device)
        elif self.param_model == "grating":
            self.theta = ckpt["theta"].to(self.device)
            self.alpha = ckpt["alpha"].to(self.device)
        else:
            raise ValueError(f"Unknown parameterization: {self.param_model}")

    def surf_dict(self):
        """Return surface parameters."""
        if self.param_model == "fresnel":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "f0": self.f0.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "binary2":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "order2": self.order2.item(),
                "order4": self.order4.item(),
                "order6": self.order6.item(),
                "order8": self.order8.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "poly1d":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "order2": self.order2.item(),
                "order3": self.order3.item(),
                "order4": self.order4.item(),
                "order5": self.order5.item(),
                "order6": self.order6.item(),
                "order7": self.order7.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "grating":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "theta": self.theta.item(),
                "alpha": self.alpha.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        return surf_dict

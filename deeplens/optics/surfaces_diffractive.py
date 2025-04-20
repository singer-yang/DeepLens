"""Diffractive optical surfaces. The input and output of each surface is a complex wave field."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .basics import DeepObj, EPSILON


# =======================================
# Diffractive optical surfaces
# =======================================
class DOE(DeepObj):
    def __init__(self, l, d, res, fab_ps=0.001, param_model="pixel2d", device="cpu"):
        """DOE class."""
        # super().__init__()
        raise Exception("surface_diffractive.py is deprecated. Use deeplens.optics.diffractive_surface instead.")

        # DOE material
        self.glass = "fused_silica"  # DOE substrate material
        self.wvln0 = 0.55  # [um], DOE design wavelength
        self.n0 = self.refractive_index(
            self.wvln0
        )  # refractive index of the substrate at design wavelength

        # DOE geometry
        self.d = torch.tensor([d]) if not isinstance(d, torch.Tensor) else d
        self.l = l
        self.r = l / 2 * float(np.sqrt(2))
        self.w = self.l
        self.h = self.l

        # DOE phase map
        self.res = [res, res] if isinstance(res, int) else res
        self.ps = (
            self.l / self.res[0]
        )  # DOE sampling pixel size (need to be small to meet Nyquist sampling condition)
        self.fab_ps = fab_ps  # default 1 [um], DOE fabrication feature size
        self.fab_res = [
            int(round(self.l / self.fab_ps)),
            int(round(self.l / self.fab_ps)),
        ]  # fabrication resolution
        assert self.res[0] % self.fab_res[0] == 0, (
            "DOE sampling resolution (to meet Nyquist criterion) should be integer times of fabrication resolution."
        )
        self.x, self.y = torch.meshgrid(
            torch.linspace(
                -self.w / 2 + self.fab_ps / 2,
                self.w / 2 - self.fab_ps / 2,
                self.fab_res[0],
            ),
            torch.linspace(
                self.h / 2 - self.fab_ps / 2,
                -self.h / 2 + self.fab_ps / 2,
                self.fab_res[1],
            ),
            indexing="xy",
        )

        self.to(device)
        self.init_param_model(param_model)

    def init_param_model(self, param_model="none", **kwargs):
        """Initialize DOE phase map.

        Contributor: Linyark

        Args:
            parameterization (str, optional): DOE parameterization method. Defaults to 'fourier'.
            mode (str, optional): DOE intialization method. Defaults to 'zero'.
            path (str, optional): DOE data path. Defaults to None.
        """
        self.param_model = param_model

        if self.param_model == "fresnel":
            # "Phase fresnel" or "Fresnel zone plate (FPZ)"
            f0 = kwargs.get("f0", 100.0)
            self.f0 = torch.tensor([f0])

            # In the future we donot want to give another wvln
            fresnel_wvln = kwargs.get("fresnel_wvln", 0.55)
            self.fresnel_wvln = fresnel_wvln

        elif self.param_model == "cubic":
            self.a3 = torch.tensor([0.001])

        elif self.param_model == "binary2":
            # Zemax binary2 phase mask
            rand_value = np.random.rand(4).astype(np.float32) * 0.01
            self.order2 = torch.tensor(rand_value[0])
            self.order4 = torch.tensor(rand_value[1])
            self.order6 = torch.tensor(rand_value[2])
            self.order8 = torch.tensor(rand_value[3])

        elif self.param_model == "binary2_fast":
            # Inverting the orders can speed up the optimization
            rand_value = (np.random.rand(4).astype(np.float32) - 0.5) * 100.0
            self.order2 = torch.tensor(rand_value[0])
            self.order4 = torch.tensor(rand_value[1])
            self.order6 = torch.tensor(rand_value[2])
            self.order8 = torch.tensor(rand_value[3])

        elif self.param_model == "poly1d":
            # Even polynomial for aberration correction, odd polynomial for EDoF.
            rand_value = np.random.rand(6).astype(np.float32) * 0.001
            self.order2 = torch.tensor(rand_value[0])
            self.order3 = torch.tensor(rand_value[1])
            self.order4 = torch.tensor(rand_value[2])
            self.order5 = torch.tensor(rand_value[3])
            self.order6 = torch.tensor(rand_value[4])
            self.order7 = torch.tensor(rand_value[5])

        elif self.param_model == "zernike":
            # Zernike polynomials, same as Zemax.
            self.z_coeff = torch.randn(37) * 0.001

        elif self.param_model == "pixel2d":
            self.pmap = torch.randn(self.res) * 0.001

        elif self.param_model == "none" or "zero":
            self.param_model = "pixel2d"
            self.pmap = torch.zeros(self.res)

        else:
            raise Exception("Unknown parameterization.")

        self.to(self.device)

    def save_doe(self, save_path="./doe.pth"):
        """Save DOE phase map."""
        self.save_ckpt(save_path)

    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE phase map."""
        if self.param_model == "fresnel":
            torch.save(
                {"param_model": self.param_model, "f0": self.f0.clone().detach().cpu()},
                save_path,
            )

        elif self.param_model == "cubic":
            torch.save(
                {"param_model": self.param_model, "a3": self.a3.clone().detach().cpu()},
                save_path,
            )

        elif self.param_model == "binary2":
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

        elif self.param_model == "binary2_fast":
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

        elif self.param_model == "zernike":
            torch.save(
                {
                    "param_model": self.param_model,
                    "z_coeff": self.z_coeff.clone().detach().cpu(),
                },
                save_path,
            )

        elif self.param_model == "pixel2d":
            torch.save(
                {
                    "param_model": self.param_model,
                    "pmap": self.pmap.clone().detach().cpu(),
                },
                save_path,
            )

        else:
            raise Exception("Unknown parameterization.")

    def load_doe(self, doe_dict):
        """Load DOE parameters from a dict."""
        # Init DOE parameter model
        param_model = doe_dict["param_model"]
        self.init_param_model(param_model)

        # Load DOE parameters
        if self.param_model == "fresnel":
            self.f0 = doe_dict["f0"].to(self.device)

        elif self.param_model == "cubic":
            self.a3 = doe_dict["a3"].to(self.device)

        elif self.param_model == "binary2":
            self.order2 = doe_dict["order2"].to(self.device)
            self.order4 = doe_dict["order4"].to(self.device)
            self.order6 = doe_dict["order6"].to(self.device)
            self.order8 = doe_dict["order8"].to(self.device)

        elif self.param_model == "binary2_fast":
            self.order2 = doe_dict["order2"].to(self.device)
            self.order4 = doe_dict["order4"].to(self.device)
            self.order6 = doe_dict["order6"].to(self.device)
            self.order8 = doe_dict["order8"].to(self.device)

        elif self.param_model == "poly1d":
            self.order2 = doe_dict["order2"].to(self.device)
            self.order3 = doe_dict["order3"].to(self.device)
            self.order4 = doe_dict["order4"].to(self.device)
            self.order5 = doe_dict["order5"].to(self.device)
            self.order6 = doe_dict["order6"].to(self.device)
            self.order7 = doe_dict["order7"].to(self.device)

        elif self.param_model == "zernike":
            self.z_coeff = doe_dict["z_coeff"].to(self.device)

        elif self.param_model == "pixel2d":
            self.pmap = doe_dict["pmap"].to(self.device)

        else:
            raise Exception("Unknown parameterization.")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE phase map."""
        ckpt = torch.load(load_path)
        self.load_doe(ckpt)

    # =======================================
    # Computation
    # =======================================
    def get_phase_map(self, wvln=0.55):
        """Calculate phase map of the DOE at the given wavelength.

        First we should calculate the phase map at 0.55um, then calculate the phase map for the given other wavelength.

        Args:
            wvln (float): Wavelength. [um]. Defaults to 0.55.

        Returns:
            phase_map (tensor): Phase map. [1, 1, H, W], range [0, 2pi].
        """
        n = self.refractive_index(wvln)
        phase_map0 = self.get_phase_map0()
        phase_map = phase_map0 * (self.wvln0 / wvln) * (n - 1) / (self.n0 - 1)

        phase_map = (
            F.interpolate(
                phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )
        return phase_map

    def get_phase_map0(self):
        """Calculate phase map at wvln 0.55 um.

        Returns:
            pmap (tensor): phase map at 0.55 um, range [0, 2pi].
        """
        x_norm = self.x / self.r
        y_norm = self.y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + 1e-12)

        if self.param_model == "fresnel":
            # unit [mm]
            pmap = (
                -2
                * math.pi
                * torch.fmod(
                    (self.x**2 + self.y**2) / (2 * self.fresnel_wvln * 1e-3 * self.f0),
                    1,
                )
            )

        elif self.param_model == "cubic":
            pmap = self.a3 * (self.x**3 + self.y**3)

        elif self.param_model == "binary2":
            pmap = (
                self.order2 * r**2
                + self.order4 * r**4
                + self.order6 * r**6
                + self.order8 * r**8
            )

        elif self.param_model == "binary2_fast":
            pmap = (
                1 / (self.order2 + EPSILON) * r**2
                + 1 / (self.order4 + EPSILON) * r**4
                + 1 / (self.order6 + EPSILON) * r**6
                + 1 / (self.order8 + EPSILON) * r**8
            )

        elif self.param_model == "poly1d":
            pmap_even = self.order2 * r**2 + self.order4 * r**4 + self.order6 * r**6
            pmap_odd = (
                self.order3 * (x_norm**3 + y_norm**3)
                + self.order5 * (x_norm**5 + y_norm**5)
                + self.order7 * (x_norm**7 + y_norm**7)
            )
            pmap = pmap_even + pmap_odd

        elif self.param_model == "zernike":
            alpha = torch.atan2(y_norm, x_norm)
            pmap = Zernike(self.z_coeff, r, alpha)

        elif self.param_model == "pixel2d":
            pmap = self.pmap

        else:
            raise Exception("Unknown parameterization.")

        pmap = torch.remainder(pmap, 2 * torch.pi)
        return pmap

    def refractive_index(self, wvln=0.55):
        """Calculate refractive index of DOE. Used for phase map calculation."""
        if self.glass == "fused_silica":
            assert wvln >= 0.4 and wvln <= 0.7, (
                "Wavelength should be in the range of [0.4, 0.7] um."
            )
            ref_wvlns = [
                0.40,
                0.41,
                0.42,
                0.43,
                0.44,
                0.45,
                0.46,
                0.47,
                0.48,
                0.49,
                0.50,
                0.51,
                0.52,
                0.53,
                0.54,
                0.55,
                0.56,
                0.57,
                0.58,
                0.59,
                0.60,
                0.61,
                0.62,
                0.63,
                0.64,
                0.65,
                0.66,
                0.67,
                0.68,
                0.69,
                0.70,
            ]

            ref_n = [
                1.4701,
                1.4692,
                1.4683,
                1.4674,
                1.4665,
                1.4656,
                1.4649,
                1.4642,
                1.4636,
                1.4629,
                1.4623,
                1.4619,
                1.4614,
                1.4610,
                1.4605,
                1.4601,
                1.4597,
                1.4593,
                1.4589,
                1.4585,
                1.4580,
                1.4577,
                1.4574,
                1.4571,
                1.4568,
                1.4565,
                1.4563,
                1.4560,
                1.4558,
                1.4555,
                1.4553,
            ]

            # Find the nearest two wvlns
            delta_wvln = [abs(wv - wvln) for wv in ref_wvlns]
            idx1 = delta_wvln.index(min(delta_wvln))
            delta_wvln[idx1] = float("inf")
            idx2 = delta_wvln.index(min(delta_wvln))

            # Interpolate n
            n = ref_n[idx1] + (ref_n[idx2] - ref_n[idx1]) / (
                ref_wvlns[idx2] - ref_wvlns[idx1]
            ) * (wvln - ref_wvlns[idx1])

        else:
            raise Exception("Unknown DOE material.")

        return n

    def pmap_quantize(self, bits=16):
        """Quantize phase map to bits levels."""
        pmap = self.get_phase_map0()
        pmap_q = torch.round(pmap / (2 * float(np.pi) / bits)) * (2 * float(np.pi) / bits)
        return pmap_q

    def pmap_fab(self, bits=16, save_path=None):
        """Convert to fabricate phase map and save it. This function is used to output DOE_fab file, and it will not change the DOE object itself."""
        # Fab resolution quantized pmap
        pmap = self.get_phase_map0()
        fab_res = int(self.ps / self.fab_ps * self.res[0])
        pmap = (
            F.interpolate(
                pmap.unsqueeze(0).unsqueeze(0),
                scale_factor=self.ps / self.fab_ps,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        pmap_q = torch.round(pmap / (2 * float(np.pi) / bits)) * (2 * float(np.pi) / bits)

        # Save phase map
        if save_path is None:
            save_path = f"./doe_fab_{fab_res}x{fab_res}_{int(self.fab_ps * 1000)}um_{bits}bit.pth"
        self.save_ckpt(save_path=save_path)

        return pmap_q

    def loss_quantization(self, bits=16):
        """DOE quantization errors.

        Reference: Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging
        """
        pmap = self.get_phase_map0()
        pmap_q = self.pmap_quantize(bits)
        loss = torch.mean(torch.abs(pmap - pmap_q))
        return loss

    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self, activate=True):
        """Activate gradient for phase map parameters."""
        if self.param_model == "fresnel":
            self.f0.requires_grad = activate

        elif self.param_model == "cubic":
            self.a3.requires_grad = activate

        elif self.param_model == "binary2":
            self.order2.requires_grad = activate
            self.order4.requires_grad = activate
            self.order6.requires_grad = activate
            self.order8.requires_grad = activate

        elif self.param_model == "binary2_fast":
            self.order2.requires_grad = activate
            self.order4.requires_grad = activate
            self.order6.requires_grad = activate
            self.order8.requires_grad = activate

        elif self.param_model == "poly1d":
            self.order2.requires_grad = activate
            self.order3.requires_grad = activate
            self.order4.requires_grad = activate
            self.order5.requires_grad = activate
            self.order6.requires_grad = activate
            self.order7.requires_grad = activate

        elif self.param_model == "zernike":
            self.z_coeff.requires_grad = activate

        elif self.param_model == "pixel2d":
            self.pmap.requires_grad = activate

        else:
            raise NotImplementedError

    def get_optimizer_params(self, lr=None):
        self.activate_grad()
        params = []

        if self.param_model == "fresnel":
            lr = 0.001 if lr is None else lr
            params.append({"params": [self.f0], "lr": lr})

        elif self.param_model == "cubic":
            lr = 0.1 if lr is None else lr
            params.append({"params": [self.a3], "lr": lr})

        elif self.param_model == "binary2":
            lr = 0.1 if lr is None else lr
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order8], "lr": lr})

        elif self.param_model == "binary2_fast":
            lr = 0.1 if lr is None else lr
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order8], "lr": lr})

        elif self.param_model == "poly1d":
            lr = 0.1 if lr is None else lr
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order3], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order5], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order7], "lr": lr})

        elif self.param_model == "zernike":
            lr = 0.01 if lr is None else lr
            params.append({"params": [self.z_coeff], "lr": lr})

        elif self.param_model == "pixel2d":
            lr = 0.01 if lr is None else lr
            params.append({"params": [self.pmap], "lr": lr})

        else:
            raise NotImplementedError

        return params

    def get_optimizer(self, lr=None):
        """Generate optimizer for DOE.

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)

        return optimizer

    def forward(self, field):
        """1, Propagate to DOE.
            2, Apply phase modulation.

            Recommaneded field has [B, 1, H, W] shape.

            Consider input field has different pixel size ad physical size with the DOE.

            Reference: https://github.com/vsitzmann/deepoptics function phaseshifts_from_height_map

        Args:
            field (Field): Input complex wave field.
        """
        # ==> 1. Propagate to DOE
        field.prop_to(self.d)

        # ==> 2. Compute and resize phase map
        phase_map = self.get_phase_map(
            field.wvln
        )  # recommanded to have [1, H, W] shape
        assert self.h == field.phy_size[0], (
            "Wave field and DOE physical should have the same physical size."
        )
        if not field.u.shape[-2:] == phase_map.shape[-2:]:
            raise Exception(
                "Field and phase map resolution should be the same. Interpolation can be done but not a desired way."
            )

        field.u = field.u * torch.exp(1j * phase_map)
        return field

    # =======================================
    # Visualization
    # =======================================
    def show(self, save_name="./DOE_phase_map.png"):
        """Visualize phase map."""
        self.draw_phase_map(save_name)

    def save_pmap(self, save_path="./DOE_phase_map.png"):
        """Save phase map."""
        self.draw_phase_map(save_path)

    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2pi]."""
        pmap = self.get_phase_map0()
        save_image(pmap, save_name, normalize=True)

    def draw_phase_map_fab(self, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2pi]."""
        pmap = self.get_phase_map0()
        pmap_q = self.pmap_quantize()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[0].set_title(f"Phase map ({self.wvln0}um)", fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])

        ax[1].imshow(pmap_q.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[1].set_title(f"Quantized phase map ({self.wvln0}um)", fontsize=10)
        ax[1].grid(False)
        fig.colorbar(ax[1].get_images()[0])

        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_phase_map3d(self, save_name="./DOE_phase_map3d.png"):
        """Draw 3D phase map."""
        pmap = self.get_phase_map0() / 20.0
        x = np.linspace(-self.w / 2, self.w / 2, self.res[0])
        y = np.linspace(-self.h / 2, self.h / 2, self.res[1])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            pmap.cpu().numpy().flatten(),
            marker=".",
            s=0.01,
            c=pmap.cpu().numpy().flatten(),
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_cross_section(self, save_name="./DOE_corss_sec.png"):
        """Draw cross section of the phase map."""
        pmap = self.get_phase_map0()
        pmap = torch.diag(pmap).cpu().numpy()
        r = np.linspace(-self.w / 2 * float(np.sqrt(2)), self.w / 2 * float(np.sqrt(2)), self.res[0])

        fig, ax = plt.subplots()
        ax.plot(r, pmap)
        ax.set_title(f"Phase map ({self.wvln0}um) cross section")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def surface(self, x, y, max_offset=0.2):
        """When drawing the lens setup, this function is called to compute the surface height.

        Here we use a fake height ONLY for drawing.
        """
        roc = self.l
        r = torch.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - torch.sqrt(1 - r**2 / roc**2))
        sag = max_offset - torch.fmod(sag, max_offset)
        return sag

    def draw_wedge(self, ax, color="black"):
        # Create radius points
        r = torch.linspace(-self.r, self.r, 256, device=self.device)
        offset = 0.1

        # Draw base at z = self.d
        base_z = torch.tensor([self.d + offset, self.d, self.d, self.d + offset])
        base_x = torch.tensor([-self.r, -self.r, self.r, self.r])
        base_points = torch.stack((base_x, torch.zeros_like(base_x), base_z), dim=-1)
        base_points = base_points.cpu().detach().numpy()
        ax.plot(base_points[..., 2], base_points[..., 0], color=color, linewidth=0.8)

        # Calculate and draw surface
        z = self.surface(r, torch.zeros_like(r), max_offset=offset) + self.d + offset
        points = torch.stack((r, torch.zeros_like(r), z), dim=-1)
        points = points.cpu().detach().numpy()
        ax.plot(points[..., 2], points[..., 0], color=color, linewidth=0.8)

    # =======================================
    # Utils
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "DOE",
            "l": round(self.l, 4),
            "d": round(self.d[0].item(), 4),
            "res": self.res,
            "fab_ps": round(self.fab_ps, 6),
            "is_square": True,
            "param_model": self.param_model,
            "doe_path": None,
        }

        if self.param_model == "fresnel":
            surf_dict["f0"] = round(self.f0.item(), 6)
        elif self.param_model == "cubic":
            surf_dict["a3"] = round(self.a3.item(), 6)
        elif self.param_model == "binary2":
            surf_dict["order2"] = round(self.order2.item(), 6)
            surf_dict["order4"] = round(self.order4.item(), 6)
            surf_dict["order6"] = round(self.order6.item(), 6)
            surf_dict["order8"] = round(self.order8.item(), 6)
        elif self.param_model == "binary2_fast":
            surf_dict["order2"] = round(self.order2.item(), 6)
            surf_dict["order4"] = round(self.order4.item(), 6)
            surf_dict["order6"] = round(self.order6.item(), 6)
            surf_dict["order8"] = round(self.order8.item(), 6)
        elif self.param_model == "poly1d":
            surf_dict["order2"] = round(self.order2.item(), 6)
            surf_dict["order3"] = round(self.order3.item(), 6)
            surf_dict["order4"] = round(self.order4.item(), 6)
            surf_dict["order5"] = round(self.order5.item(), 6)
            surf_dict["order6"] = round(self.order6.item(), 6)
            surf_dict["order7"] = round(self.order7.item(), 6)
        elif self.param_model == "zernike":
            surf_dict["z_coeff"] = self.z_coeff.tolist()
        elif self.param_model == "pixel2d":
            raise NotImplementedError
            surf_dict["pmap"] = self.pmap.tolist()
        else:
            raise NotImplementedError

        return surf_dict


class ThinLens(DeepObj):
    """Paraxial optical model for refractive lens.

    Two types of thin lenses supported:
        (1) Thin lens without chromatic aberration.
        (2) Extended thin lens with chromatic aberration.
    """

    def __init__(self, foclen, d, r):
        super().__init__()

        self.foclen = foclen
        self.d = torch.tensor([d])
        self.r = float(r)
        self.chromatic = False

    def load_foclens(self, wvlns, foclens):
        """Load a list of wvlns and corresponding focus lenghs for interpolation. This function is used for chromatic aberration simulation."""
        self.chromatic = True
        self.ref_wvlns = wvlns
        self.ref_foclens = foclens

    def interp_foclen(self, wvln):
        """Interpolate focus length for different wvln."""
        assert wvln > 0.1 and wvln < 1.0, "Wavelength unit should be [um]."
        ref_wvlns = self.ref_wvlns
        ref_foclens = self.ref_foclens

        # Find the nearest two wvlns
        delta_wvln = [abs(wv - wvln) for wv in ref_wvlns]
        idx1 = delta_wvln.index(min(delta_wvln))
        delta_wvln[idx1] = float("inf")
        idx2 = delta_wvln.index(min(delta_wvln))

        # Interpolate focus length
        foclen = ref_foclens[idx1] + (ref_foclens[idx2] - ref_foclens[idx1]) / (
            ref_wvlns[idx2] - ref_wvlns[idx1]
        ) * (wvln - ref_wvlns[idx1])
        return foclen

    def forward(self, field):
        """Thin lens propagation.

        1, propagate the wave to the lens plane
        2, apply phase change and aperture
        """
        # ==> Propagate to the lens plane
        field.prop_to(z=self.d)

        # ==> Apply phase change and aperture
        foclen = self.interp_foclen(field.wvln) if self.chromatic else self.foclen
        phi_lens = torch.fmod(
            -field.k * (field.x**2 + field.y**2) / (2 * foclen), 2 * float(np.pi)
        )
        field.u *= torch.exp(1j * phi_lens)

        # ==> Apply aperture
        aper_lens = torch.sqrt(field.x**2 + field.y**2) < self.r
        field.u *= aper_lens

        return field

    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "ThinLens",
            "foclen": float(f"{self.foclen:.6f}"),
            "r": float(f"{self.r:.6f}"),
        }
        return surf_dict


class Aperture(DeepObj):
    def __init__(self, d, r, device="cpu"):
        super().__init__()
        self.d = torch.tensor([d])
        self.r = r

    def forward(self, field):
        """Propagate the wave to the aperture plane and apply aperture."""
        field.prop_to(z=self.d)
        aper_lens = torch.sqrt(field.x**2 + field.y**2) < self.r
        field.u *= aper_lens

        return field

    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "Aperture",
            "r": float(f"{self.r:.6f}"),
        }
        return surf_dict


class Sensor(DeepObj):
    def __init__(self, d, r=None, l=None, res=[2048, 2048], device="cpu"):
        """Image sensor class. Now only square sensor is considered.

        Args:
            d (float): Global position.
            r (float): Half diagonal distance.
        """
        super(Sensor, self).__init__()
        res = [res, res] if isinstance(res, int) else res
        self.d = torch.tensor([d])
        if r is None and l is None:
            raise Exception("Either r or l should be specified.")
        if r is not None:
            self.r = r
            self.l = r * math.sqrt(2)
        if l is not None:
            self.l = l
            self.r = l / 2 * math.sqrt(2)

        self.res = res
        self.ps = self.l / self.res[0]  # pixel size

        self.to(device)

    def forward(self, field):
        """Propagate a field to the sensor. Output the sensor intensity response to the field.

        Return:
            response (tensor): Sensor intensity response. Shape of [N, C, H, W].
        """
        # Propagate to sensor plane
        field.prop_to(self.d)

        # Energy response
        assert len(field.u.shape) == 4, "Input field should have shape of [N, C, H, W]."
        response = field.u.abs() ** 2

        # If resolutions are different, we resize the response
        if response.shape[-1] != self.res[-1]:
            response = F.interpolate(
                response, self.res, mode="bilinear", align_corners=False
            )

        return response

    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "Sensor",
            "res": self.res,
            "l": float(f"{self.l:.6f}"),
        }
        return surf_dict


# =======================================
# Functions
# =======================================
def Zernike(z_coeff, grid=256):
    """Calculate phase map produced by the first 37 Zernike polynomials. The output zernike phase map is in real value, to use it in the future we need to convert it to complex value."""
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

    # Mask out
    mask = torch.gt(x**2 + y**2, 1)
    ZW[mask] = 0.0

    return ZW

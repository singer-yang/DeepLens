"""Base class for diffractive surfaces. Here we assume all diffractive surfaces are DOE/Multi-layer optical elemnents."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from ..basics import EPSILON, DeepObj
from ..materials import Material


class DiffractiveSurface(DeepObj):
    def __init__(self, d, size, res=(2000, 2000), wvln0=0.55, mat="fused_silica", fab_ps=0.001, device="cpu"):
        """
        Args:
            d (float): Distance of the DOE surface. [mm]
            size (tuple or int): Size of the DOE, [w, h]. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            mat (str): Material of the DOE.
            fab_ps (float): Fabrication pixel size. [mm]
            device (str): Device to run the DOE.
        """
        # Geometry
        self.d = torch.tensor(d) if not isinstance(d, torch.Tensor) else d
        if isinstance(size, int) or isinstance(size, float):
            self.size = [size, size]
        else:
            self.size = size
        self.w = self.size[0]
        self.h = self.size[1]
        self.res = [res, res] if isinstance(res, int) else res
        self.ps = self.w / self.res[0] # pixel size

        # Phase map
        self.mat = Material(mat)
        self.wvln0 = wvln0  # [um], design wavelength. Sometimes the maximum working wavelength is preferred.
        self.n0 = self.mat.refractive_index(
            self.wvln0
        )  # refractive index at design wavelength

        # Fabrication for DOE
        self.fab_ps = fab_ps # [mm], fabrication pixel size
        self.fab_step = 16

        # x, y coordinates
        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w/2, self.w/2, self.res[1]),
            torch.linspace(self.h/2, -self.h/2, self.res[0]),
            indexing="xy",
        )

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize DOE from a dict."""
        raise NotImplementedError

    def _phase_map0(self):
        """Calculate phase map at design wavelength.

        Returns:
            phase0 (tensor): phase map at design wavelength, range [0, 2pi].
        """
        raise NotImplementedError
    
    def get_phase_map0(self):
        """Calculate phase map at design wavelength.

        Returns:
            phase0 (tensor): phase map at design wavelength, range [0, 2pi].
        """
        phase0 = self._phase_map0()
        phase0 = torch.remainder(phase0, 2 * torch.pi)
        return phase0
    
    def get_phase_map(self, wvln=0.55):
        """Calculate phase map at the given wavelength.

        Args:
            wvln (float): Wavelength. [um].

        Returns:
            phase_map (tensor): Phase map. [1, 1, H, W], range [0, 2pi].

        Note:
            First we should calculate the phase map at 0.55um, then calculate the phase map for the given other wavelength.
        """
        # Phase map at design wavelength
        phase_map0 = self.get_phase_map0()
        
        # Phase map at given wavelength
        n = self.mat.refractive_index(wvln)
        phase_map = phase_map0 * (self.wvln0 / wvln) * (n - 1) / (self.n0 - 1)

        # Interpolate to the desired resolution
        phase_map = (
            F.interpolate(
                phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        return phase_map

    def forward(self, wave):
        """1, Propagate to DOE.
            2, Apply phase modulation.

            Recommaneded field has [B, 1, H, W] shape.

            Consider input field has different pixel size ad physical size with the DOE.

            Reference: https://github.com/vsitzmann/deepoptics function phaseshifts_from_height_map

        Args:
            wave (Wave): Input complex wave field.
        """
        # ==> 1. Propagate to DOE
        wave.prop_to(self.d)

        # ==> 2. Compute and resize phase map
        phase_map = self.get_phase_map(
            wave.wvln
        )  # recommanded to have [1, H, W] shape
        assert self.h == wave.phy_size[0], (
            "Wave field and DOE physical should have the same physical size."
        )
        if not wave.u.shape[-2:] == phase_map.shape[-2:]:
            raise Exception(
                "Field and phase map resolution should be the same. Interpolation can be done but not a desired way."
            )

        wave.u = wave.u * torch.exp(1j * phase_map)
        return wave
    
    def __call__(self, wave):
        """Forward function.
        
        Args:
            wave (Wave): Input complex wave field.

        Returns:
            wave (Wave): Output complex wave field.
        """
        return self.forward(wave)

    # =======================================
    # Fabrication-related functions
    # =======================================
    def pmap_quantize(self, bits=16):
        """Quantize phase map to bits levels."""
        pmap = self.get_phase_map0()
        pmap_q = torch.round(pmap / (2 * float(np.pi) / bits)) * (
            2 * float(np.pi) / bits
        )
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
        pmap_q = torch.round(pmap / (2 * float(np.pi) / bits)) * (
            2 * float(np.pi) / bits
        )

        # Save phase map
        if save_path is None:
            save_path = f"./doe_fab_{fab_res}x{fab_res}_{int(self.fab_ps * 1000)}um_{bits}bit.pth"
        self.save_ckpt(save_path=save_path)

        return pmap_q

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

    def loss_quantization(self, bits=16):
        """DOE quantization errors.

        Reference: Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging
        """
        pmap = self.get_phase_map0()
        pmap_q = self.pmap_quantize(bits)
        loss = torch.mean(torch.abs(pmap - pmap_q))
        return loss

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

    def draw_cross_section(self, save_name="./DOE_cross_section.png"):
        """Draw cross section of the phase map."""
        pmap = self.get_phase_map0()
        pmap = torch.diag(pmap).cpu().numpy()
        r = np.linspace(
            -self.w / 2 * float(np.sqrt(2)), self.w / 2 * float(np.sqrt(2)), self.res[0]
        )

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
        """Draw 2d wedge in the plot."""
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
            "type": self.__class__.__name__,
            "size": [round(self.w, 4), round(self.h, 4)],
            "d": round(self.d.item(), 4),
            "wvln0": round(self.wvln0, 4),
            "res": self.res,
            "is_square": True,
        }

        return surf_dict

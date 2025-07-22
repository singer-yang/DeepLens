"""Base class for diffractive surfaces (DOE)."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from deeplens.optics.basics import EPSILON, DeepObj
from deeplens.optics.materials import Material


class DiffractiveSurface(DeepObj):
    def __init__(
        self,
        d,
        res=(2000, 2000),
        fab_ps=0.001,
        wvln0=0.55,
        mat="fused_silica",
        design_ps=None,
        device="cpu",
    ):
        """Diffractive surface class. Optical properties are simulated with wave optics.
        
        Args:
            d (float): Distance of the DOE surface. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            fab_ps (float): Fabrication pixel size. [mm]
            wvln0 (float): Design wavelength. [um]
            mat (str): Material of the DOE.
            design_ps (float): Design pixel size. [mm]
            device (str): Device to run the DOE.
        """
        # Geometry
        self.d = torch.tensor(d) if not isinstance(d, torch.Tensor) else d
        self.res = [res, res] if isinstance(res, int) else res
        self.ps = fab_ps if design_ps is None else design_ps
        self.w = self.res[0] * self.ps
        self.h = self.res[1] * self.ps

        # Phase map
        self.mat = Material(mat)
        self.wvln0 = wvln0  # [um], design wavelength. Sometimes the maximum working wavelength is preferred.
        self.n0 = self.mat.refractive_index(
            self.wvln0
        )  # refractive index at design wavelength

        # Fabrication for DOE
        self.fab_ps = fab_ps  # [mm], fabrication pixel size
        self.fab_step = 16

        # x, y coordinates
        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
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
        """Propagate wave field to the DOE and apply phase modulation. Input wave field can have different pixel size and physical size with the DOE.

        Args:
            wave (Wave): Input complex wave field. Shape of [B, 1, H, W].

        Returns:
            wave (Wave): Output complex wave field. Shape of [B, 1, H, W].

        Reference:
            [1] https://github.com/vsitzmann/deepoptics function phaseshifts_from_height_map
        """
        # Propagate to DOE
        wave.prop_to(self.d)

        # Compute phase map at the wave field wavelength, shape of [H, W]
        phase_map = self.get_phase_map(wave.wvln)

        # Consider the different pixel size between the wave field and the DOE
        if self.ps != wave.ps:
            scale = self.ps / wave.ps
            phase_map = (
                F.interpolate(
                    phase_map.unsqueeze(0).unsqueeze(0),
                    scale_factor=(scale, scale),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        # Check if the field and phase map resolution (physical size) are the same
        wave_h, wave_w = wave.u.shape[-2:]
        phase_h, phase_w = phase_map.shape[-2:]
        if phase_h > wave_h or phase_w > wave_w:
            start_h = (phase_h - wave_h) // 2
            start_w = (phase_w - wave_w) // 2
            phase_map = phase_map[
                ..., start_h : start_h + wave_h, start_w : start_w + wave_w
            ]
        elif phase_h < wave_h or phase_w < wave_w:
            pad_top = (wave_h - phase_h) // 2
            pad_bottom = wave_h - phase_h - pad_top
            pad_left = (wave_w - phase_w) // 2
            pad_right = wave_w - phase_w - pad_left
            phase_map = F.pad(
                phase_map,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
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
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)
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
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)

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
        raise NotImplementedError

    def get_optimizer_params(self, lr=None):
        raise NotImplementedError

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

    def draw_phase_map_nbit(self, bits=16, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2pi]."""
        pmap_q = self.pmap_quantize(bits)
        save_image(pmap_q, save_name, normalize=True)

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

    def draw_quantized_phase_map3d(self, bits=16, save_name="./DOE_quantized_phase_map3d.png"):
        """Draw 3D quantized phase map."""
        pmap = self.pmap_quantize(bits) / 20.0
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

    def draw_widget(self, ax, color="black"):
        """Draw 2d widget in the plot."""
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

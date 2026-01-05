# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Complex wave field class for diffraction simulation.

This file contains:
    1. Complex wave field class
    2. Wave field propagation functions (ASM, Rayleigh Sommerfeld, Fresnel, Fraunhofer, etc.)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens.basics import DELTA, DeepObj


# ===================================
# Complex wave field
# ===================================
class ComplexWave(DeepObj):
    def __init__(
        self,
        u=None,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
    ):
        """Complex wave field.

        Args:
            u (tensor): complex wave field, shape [H, W] or [B, C, H, W].
            wvln (float): wavelength in [um].
            z (float): distance in [mm].
            phy_size (tuple): physical size in [mm].
            res (tuple): resolution.
        """
        if u is not None:
            if not u.dtype == torch.complex128:
                print(
                    "A complex wave field is created with single precision. In the future, we want to always use double precision."
                )

            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)

            # [H, W] or [1, H, W] to [1, 1, H, W]
            if len(u.shape) == 2:
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:
                self.u = self.u.unsqueeze(0)

            self.res = self.u.shape[-2:]

        else:
            # Initialize a zero complex wave field
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res

        # Wave field parameters
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        self.wvln = wvln  # [um], wavelength
        self.k = 2 * torch.pi / (self.wvln * 1e-3)  # [mm^-1], wave number
        self.phy_size = phy_size  # [mm], physical size
        assert phy_size[0] / self.res[0] == phy_size[1] / self.res[1], (
            "Pixel size is not square."
        )
        self.ps = phy_size[0] / self.res[0]  # [mm], pixel size

        # Wave field grid
        self.x, self.y = self.gen_xy_grid()  # x, y grid
        self.z = torch.full_like(self.x, z)  # z grid

    @classmethod
    def point_wave(
        cls,
        point=(0, 0, -1000.0),
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        valid_r=None,
    ):
        """Create a spherical wave field on x0y plane originating from a point source.

        Args:
            point (tuple): Point source position in object space. [mm]. Defaults to (0, 0, -1000.0).
            wvln (float): Wavelength. [um]. Defaults to 0.55.
            z (float): Field z position. [mm]. Defaults to 0.0.
            phy_size (tuple): Valid plane on x0y plane. [mm]. Defaults to (2, 2).
            res (tuple): Valid plane resoltution. Defaults to (1000, 1000).
            valid_r (float): Valid circle radius. [mm]. Defaults to None.

        Returns:
            field (ComplexWave): Complex field on x0y plane.
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        k = 2 * torch.pi / (wvln * 1e-3)  # [mm^-1], wave number

        # Create meshgrid on target plane
        x, y = torch.meshgrid(
            torch.linspace(
                -0.5 * phy_size[0], 0.5 * phy_size[0], res[0], dtype=torch.float64
            ),
            torch.linspace(
                0.5 * phy_size[1], -0.5 * phy_size[1], res[1], dtype=torch.float64
            ),
            indexing="xy",
        )

        # Calculate distance to point source, and calculate spherical wave phase
        r = torch.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2)
        if point[2] < z:
            phi = k * r
        else:
            phi = -k * r
        u = (r.min() / r) * torch.exp(1j * phi)

        # Apply valid circle if provided, e.g., the aperture of a lens
        if valid_r is not None:
            mask = (x - point[0]) ** 2 + (y - point[1]) ** 2 < valid_r**2
            u = u * mask

        # Create wave field
        return cls(u=u, wvln=wvln, phy_size=phy_size, res=res, z=z)

    @classmethod
    def plane_wave(
        cls,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        valid_r=None,
    ):
        """Create a planar wave field on x0y plane.

        Args:
            wvln (float): Wavelength. [um].
            z (float): Field z position. [mm].
            phy_size (tuple): Physical size of the field. [mm].
            res (tuple): Resolution.
            valid_r (float): Valid circle radius. [mm].

        Returns:
            field (ComplexWave): Complex field.
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."

        # Create a plane wave field
        u = torch.ones(res, dtype=torch.float64) + 0j

        # Apply valid circle if provided
        if valid_r is not None:
            x, y = torch.meshgrid(
                torch.linspace(-0.5 * phy_size[0], 0.5 * phy_size[0], res[0]),
                torch.linspace(-0.5 * phy_size[1], 0.5 * phy_size[1], res[1]),
                indexing="xy",
            )
            mask = (x**2 + y**2) < valid_r**2
            u = u * mask

        # Create wave field
        return cls(u=u, phy_size=phy_size, wvln=wvln, res=res, z=z)

    @classmethod
    def image_wave(cls, img, wvln=0.55, z=0.0, phy_size=(4.0, 4.0)):
        """Initialize a complex wave field from an image.

        Args:
            img (torch.Tensor): Input image with shape [H, W] or [B, C, H, W]. Data range is [0, 1].
            wvln (float): Wavelength. [um].
            z (float): Field z position. [mm].
            phy_size (tuple): Physical size of the field. [mm].

        Returns:
            field (ComplexWave): Complex field.
        """
        assert img.dtype == torch.float32, "Image must be float32."

        amp = torch.sqrt(img)
        phi = torch.zeros_like(amp)
        u = amp + 1j * phi

        return cls(u=u, wvln=wvln, phy_size=phy_size, res=u.shape[-2:], z=z)

    # =============================================
    # Wave propagation
    # =============================================
    def prop(self, prop_dist, n=1.0):
        """Propagate the field by distance z. Can only propagate planar wave.

        Reference:
            [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Table 1.
            [2] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py
            [3] https://spie.org/samples/PM103.pdf
            [4] "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"

        Args:
            prop_dist (float): propagation distance, unit [mm].
            n (float): refractive index.

        Returns:
            self: propagated complex wave field.
        """
        # Determine propagation method and perform propagation
        wvln_mm = self.wvln * 1e-3  # [um] to [mm]
        asm_zmax = Nyquist_ASM_zmax(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])
        fresnel_zmin = Fresnel_zmin(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])
        
        # Wave propagation methods
        if prop_dist < DELTA:
            # Zero distance: do nothing
            pass
        
        elif prop_dist < wvln_mm:
            # Sub-wavelength distance: full wave method (e.g., FDTD)
            raise Exception(
                "The propagation distance in sub-wavelength range is not implemented yet. Have to use full wave method (e.g., FDTD)."
            )
        
        elif prop_dist < asm_zmax:
            # Angular Spectrum Method (ASM)
            self.u = AngularSpectrumMethod(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        elif prop_dist > fresnel_zmin:
            # Fresnel diffraction
            self.u = FresnelDiffraction(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        else:
            raise Exception(f"Propagation method not implemented for distance {prop_dist} mm.")
        
        # Update z grid
        self.z += prop_dist
        return self

    def prop_to(self, z, n=1):
        """Propagate the field to plane z.

        Args:
            z (float): destination plane z coordinate.
        """
        prop_dist = z - self.z[0, 0].item()
        self.prop(prop_dist, n=n)
        return self

    # =============================================
    # Helper functions
    # =============================================

    def gen_xy_grid(self):
        """Generate the x and y grid."""
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[1], 0.5 * self.phy_size[1], self.res[0],),
            torch.linspace(0.5 * self.phy_size[0], -0.5 * self.phy_size[0], self.res[1],),
            indexing="xy",
        )
        return x, y

    def gen_freq_grid(self):
        """Generate the frequency grid."""
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy

    # =============================================
    # Wave field I/O
    # =============================================

    def load(self, filepath):
        if filepath.endswith(".npz"):
            self.load_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def load_npz(self, filepath):
        """Load data from npz file."""
        data = np.load(filepath)
        self.u = torch.from_numpy(data["u"])
        self.x = torch.from_numpy(data["x"])
        self.y = torch.from_numpy(data["y"])
        self.wvln = data["wvln"].item()
        self.phy_size = data["phy_size"].tolist()
        self.res = self.u.shape[-2:]

    def save(self, filepath="./wavefield.npz"):
        """Save the complex wave field to a npz file."""
        if filepath.endswith(".npz"):
            self.save_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def save_npz(self, filepath="./wavefield.npz"):
        """Save the complex wave field to a npz file."""
        # Save data
        np.savez_compressed(
            filepath,
            u=self.u.cpu().numpy(),
            x=self.x.cpu().numpy(),
            y=self.y.cpu().numpy(),
            wvln=np.array(self.wvln),
            phy_size=np.array(self.phy_size),
        )

        # Save intensity, amplitude, and phase images
        u = self.u.cpu()
        save_image(u.abs() ** 2, f"{filepath[:-4]}_intensity.png", normalize=True)
        save_image(u.abs(), f"{filepath[:-4]}_amp.png", normalize=True)
        save_image(u.angle(), f"{filepath[:-4]}_phase.png", normalize=True)

    def save_image(self, save_name=None, data="irr"):
        return self.show(save_name=save_name, data=data)

    def show(self, save_name=None, data="irr"):
        """Save the field as an image."""
        cmap = "gray"
        if data == "irr":
            value = self.u.detach().abs() ** 2
        elif data == "amp":
            value = self.u.detach().abs()
        elif data == "phi" or data == "phase":
            value = torch.angle(self.u).detach()
            cmap = "hsv"
        elif data == "real":
            value = self.u.real.detach()
        elif data == "imag":
            value = self.u.imag.detach()
        else:
            raise Exception(f"Unimplemented visualization: {data}.")

        if len(self.u.shape) == 2:
            raise Exception("Deprecated.")
            if save_name is not None:
                save_image(value, save_name, normalize=True)
            else:
                value = value.cpu().numpy()
                plt.imshow(
                    value,
                    cmap=cmap,
                    extent=[
                        -self.phy_size[0] / 2,
                        self.phy_size[0] / 2,
                        -self.phy_size[1] / 2,
                        self.phy_size[1] / 2,
                    ],
                )

        elif len(self.u.shape) == 4:
            B, C, H, W = self.u.shape
            if B == 1:
                if save_name is not None:
                    save_image(value, save_name, normalize=True)
                else:
                    value = value.cpu().numpy()
                    plt.imshow(
                        value[0, 0, :, :],
                        cmap=cmap,
                        extent=[
                            -self.phy_size[0] / 2,
                            self.phy_size[0] / 2,
                            -self.phy_size[1] / 2,
                            self.phy_size[1] / 2,
                        ],
                    )
            else:
                if save_name is not None:
                    plt.savefig(save_name)
                else:
                    value = value.cpu().numpy()
                    fig, axs = plt.subplots(1, B)
                    for i in range(B):
                        axs[i].imshow(
                            value[i, 0, :, :],
                            cmap=cmap,
                            extent=[
                                -self.phy_size[0] / 2,
                                self.phy_size[0] / 2,
                                -self.phy_size[1] / 2,
                                self.phy_size[1] / 2,
                            ],
                        )
                    fig.show()
        else:
            raise Exception("Unsupported complex field shape.")

    def pad(self, Hpad, Wpad):
        """Pad the input field by (Hpad, Hpad, Wpad, Wpad). This step will also expand physical size of the field.

        Args:
            Hpad (int): Number of pixels to pad on the top and bottom.
            Wpad (int): Number of pixels to pad on the left and right.

        Returns:
            self: Padded complex wave field.
        """
        self.u = F.pad(self.u, (Hpad, Hpad, Wpad, Wpad), mode="constant", value=0)

        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [
            self.phy_size[0] * self.res[0] / Horg,
            self.phy_size[1] * self.res[1] / Worg,
        ]
        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(self.x, self.z[0, 0].item())

    def flip(self):
        """Flip the field horizontally and vertically."""
        self.u = torch.flip(self.u, [-1, -2])
        self.x = torch.flip(self.x, [-1, -2])
        self.y = torch.flip(self.y, [-1, -2])
        self.z = torch.flip(self.z, [-1, -2])
        return self


# ===================================
# Diffraction functions
# ===================================
def AngularSpectrumMethod(u, z, wvln, ps, n=1.0, padding=True):
    """Angular spectrum method.

    Args:
        u (tesor): complex field, shape [H, W] or [B, 1, H, W]
        z (float): propagation distance in [mm]
        wvln (float): wavelength in [um]
        ps (float): pixel size in [mm]
        n (float): refractive index
        padding (bool): padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]

    Reference:
        [1] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py#L293
        [2] https://blog.csdn.net/zhenpixiaoyang/article/details/111569495
    """
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3 / n # [um] to [mm]
    k = 2 * torch.pi / wvln_mm  # [mm]-1

    # Shape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # Padding
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # Propagation with angular spectrum method
    fx_1d = torch.fft.fftfreq(Wimg, d=ps, device=u.device)
    fy_1d = torch.fft.fftfreq(Himg, d=ps, device=u.device)
    fx, fy = torch.meshgrid(fx_1d, fy_1d, indexing="xy")
    square_root = torch.sqrt(1 - wvln_mm**2 * (fx**2 + fy**2))
    
    # H is defined on the unshifted frequency grid to match fft2(u)
    H = torch.exp(1j * k * z * square_root)

    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifft2(fft2(u) * H)

    # Remove padding
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def ScalableASM(u, z, wvln, ps, n=1.0, padding=True):
    """Scalable angular spectrum method.

    "ScalableASM allows for propagation models where the destination pixel pitch is larger than the source pixel pitch." Optica 2023.

    Reference:
        [1] Scalable angular spectrum propagation. Optica 2023.
    """
    pass


def FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None):
    """Fresnel propagation with FFT.

    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        z (float): propagation distance
        wvln (float): wavelength in [um]
        ps (float): pixel size
        n (float): refractive index
        padding (bool): padding or not
        TF (bool): transfer function or impulse response

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.1
        [2] https://qiweb.tudelft.nl/aoi/wavefielddiffraction/wavefielddiffraction.html
        [3] https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m
    """
    # Padding
    if padding:
        try:
            _, _, Worg, Horg = u.shape
        except Exception:
            Horg, Worg = u.shape
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        _, _, Wimg, Himg = u.shape

    # Wave field parameters in medium
    assert wvln > 0.1 and wvln < 10.0, "wvln should be in [um]."
    wvln_mm = wvln / n * 1e-3  # [um] to [mm]
    k = 2 * torch.pi / wvln_mm

    # Compute x, y, fx, fy
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg, device=u.device),
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg, device=u.device),
        indexing="xy",
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5 / ps, 0.5 / ps, Wimg, device=u.device),
        torch.linspace(0.5 / ps, -0.5 / ps, Himg, device=u.device),
        indexing="xy",
    )

    # TF or IR method
    if TF is None:
        if ps > wvln_mm * np.abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False

    if TF:
        H = torch.exp(-1j * torch.pi * wvln_mm * z * (fx**2 + fy**2))
        H = fftshift(H)
    else:
        h_amp = 1 / (1j * wvln_mm * z)
        h_const_phase = torch.exp(1j * k * z)
        h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x**2 + y**2))
        h = h_const_phase * h_amp * h_phase
        H = fft2(fftshift(h)) * ps**2

    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    return u


def FraunhoferDiffraction(u, z, wvln, ps, n=1.0, padding=True):
    """Fraunhofer diffraction.

    Args:
        u: complex field, shape [H, W] or [B, 1, H, W]
        z: propagation distance
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        n: refractive index
        padding: padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.5.
    """
    # Padding
    if padding:
        Worg, Horg = u.shape
        Wpad, Hpad = Worg // 4, Horg // 4
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Wimg, Himg = u.shape

    # side length
    wvln_mm = wvln / n * 1e-3  # [um] to [mm]
    k = 2 * torch.pi / wvln_mm

    # Compute x, y, fx, fy
    L2 = wvln_mm * z / ps
    x2, y2 = torch.meshgrid(
        torch.linspace(-L2 / 2, L2 / 2, Wimg, device=u.device),
        torch.linspace(-L2 / 2, L2 / 2, Himg, device=u.device),
        indexing="xy",
    )

    # Shorter propagation will not affect final result
    h_amp = 1 / (1j * wvln_mm * z)
    h_const_phase = torch.exp(1j * k * z)
    h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x2**2 + y2**2))
    h = h_amp * h_const_phase * h_phase
    u = h * ps**2 * ifftshift(fft2(fftshift(u)))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    return u


def RayleighSommerfeld(u, z, wvln, ps, n=1.0, memory_saving=True):
    """Rayleigh-Sommerfeld diffraction.

    This function is differentiable but we donot want to use it for optimization, because it is too expensive.

    Args:
        u: complex field, shape [H, W] or [B, 1, H, W]
        z: propagation distance
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        n: refractive index
        memory_saving: memory saving

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]
    """
    _, _, H, W = u.shape
    x, y = torch.meshgrid(
        torch.linspace(
            -0.5 * W * ps + 0.5 * ps, 0.5 * W * ps - 0.5 * ps, W, device=u.device
        ),
        torch.linspace(
            0.5 * W * ps - 0.5 * ps, -0.5 * W * ps + 0.5 * ps, H, device=u.device
        ),
        indexing="xy",
    )

    if u.ndim == 2:
        u2 = RayleighSommerfeldIntegral(
            u, x1=x, y1=y, z=z, wvln=wvln, n=n, memory_saving=memory_saving
        )
    elif u.ndim == 4:
        u2 = torch.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                u2[i, j] = RayleighSommerfeldIntegral(
                    u[i, j],
                    x1=x,
                    y1=y,
                    z=z,
                    wvln=wvln,
                    n=n,
                    memory_saving=memory_saving,
                )
    return u2


def RayleighSommerfeldIntegral(
    u1, x1, y1, z, wvln, x2=None, y2=None, n=1.0, memory_saving=False
):
    """Discrete Rayleigh-Sommerfeld diffraction integration. Rayleigh-Sommerfeld diffraction is a brute force integration approach, it doesnot require any approximation. It usually works as the ground truth.

    Args:
        u1: complex amplitude of input field, shape [H1, W1]
        x1: physical coordinate of input field, unit [mm], shape [H1, W1]
        y1: physical coordinate of input field, unit [mm], shape [H1, W1]
        z: propagation distance, unit [mm]
        wvln: wavelength, unit [um]
        x2: physical coordinate of output field, unit [mm], shape [H2, W2]
        y2: physical coordinate of output field, unit [mm], shape [H2, W2]
        n: refractive index
        memory_saving: memory saving or not

    Returns:
        u2: complex amplitude of output field, shape [H2, W2]

    Reference:
        [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Eq (9).
        [2] https://www.mathworks.com/matlabcentral/fileexchange/75049-complete-rayleigh-sommerfeld-model-version-2
    """
    # Parameters
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3  # [um] to [mm]
    k = n * 2 * torch.pi / wvln_mm  # wave number [mm]-1
    if x2 is None:
        x2 = x1.clone()
    if y2 is None:
        y2 = y1.clone()

    # Nyquist sampling criterion
    max_side_dist = max(abs(x1.max() - x2.min()), abs(x2.max() - x1.min()))
    ps = (x1.max() - x1.min()) / x1.shape[-1]
    zmin = Fresnel_zmin(
        wvln=wvln, ps=ps.item(), side_length=max_side_dist.item(), n=n
    )
    assert zmin < z, (
        f"Propagation distance is too short, minimum distance is {zmin} mm."
    )

    # Rayleigh-Sommerfeld diffraction integral
    if not memory_saving:
        # Naive computation

        # Broadcast to [H1, W1, H2, W2] for tensor parallel computation
        x1 = x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x2.shape[0], x2.shape[1])
        y1 = y1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y2.shape[0], y2.shape[1])
        u1 = u1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x2.shape[0], x2.shape[1])

        # Rayleigh-Sommerfeld diffraction integral
        r2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 + z**2  # shape of [H1, W1, H2, W2]
        r = torch.sqrt(r2)
        obliq = z / r

        u2 = torch.sum(
            u1 * obliq / r * torch.exp(1j * torch.fmod(k * r, 2 * torch.pi)),
            (0, 1),
        )
        u2 = u2 / (1j * wvln_mm)

    else:
        # Patch computation
        u2 = torch.zeros_like(u1) + 0j

        # Broadcast to [H1, W1, patch_size, patch_size] for tensor parallel computation
        patch_size = 4
        x1 = x1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, patch_size, patch_size)
        y1 = y1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, patch_size, patch_size)
        u1 = u1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, patch_size, patch_size)

        # Patch computation
        for i in tqdm(range(0, x2.shape[0], patch_size)):
            for j in range(0, x2.shape[1], patch_size):
                # Target patch
                x2_patch = x2[i : i + patch_size, j : j + patch_size]
                y2_patch = y2[i : i + patch_size, j : j + patch_size]
                r2 = (x2_patch - x1) ** 2 + (y2_patch - y1) ** 2 + z**2
                r = torch.sqrt(r2)
                obliq = z / r

                # Shape of [patch_size, patch_size]
                u2_patch = torch.sum(
                    u1 * obliq / r * torch.exp(1j * torch.fmod(k * r, 2 * torch.pi)),
                    (0, 1),
                )

                # Assign to output field
                u2[i : i + patch_size, j : j + patch_size] = u2_patch

        u2 = u2 / (1j * wvln_mm)

    return u2


# ==============================
# Helper functions
# ==============================
def Nyquist_ASM_zmax(wvln, ps, side_length, n=1.0):
    """Maximum propagation distance for Angular Spectrum Method by Nyquist sampling criterion.
    
    Args:
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        side_length: side length of the field in [mm]
        n: refractive index
    """
    wvln_mm = wvln * 1e-3
    zmax = side_length * ps * n / wvln_mm
    return zmax

def Fresnel_zmin(wvln, ps, side_length, n=1.0):
    """Minimum propagation distance for Fresnel diffraction by Nyquist sampling criterion.
    
    Args:
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        side_length: side length of the field in [mm]
        n: refractive index
    """
    wvln_mm = wvln * 1e-3
    zmin = float(np.sqrt(side_length**2) / (wvln_mm / n))
    return zmin
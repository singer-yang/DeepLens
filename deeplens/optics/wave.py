# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Complex wave field class for diffraction simulation. Better to use float64 precision.

1. Complex wave field
2. Propagation functions
3. Helper functions
"""

import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torchvision.utils import save_image

from deeplens.optics.basics import DELTA, DeepObj


# ===================================
# Complex wave field
# ===================================
class ComplexWave(DeepObj):
    def __init__(
        self,
        u=None,
        wvln=0.55,
        z=0.0,
        phy_size=[4.0, 4.0],
        res=[1000, 1000],
        valid_phy_size=None,
    ):
        """Complex wave field class.

        Args:
            u (tensor): complex wave field, shape [H, W] or [B, C, H, W].
            wvln (float): wavelength in [um].
            z (float): distance in [mm].
            phy_size (list): physical size in [mm].
            valid_phy_size (list): valid physical size in [mm].
            res (list): resolution.
        """
        # Create a complex wave field with [N, 1, H, W] shape for batch processing
        if u is not None:
            # Initialize a complex wave field with given complex amplitude
            if not u.dtype == torch.complex128:
                print(
                    "A complex wave field is created with single precision. In the future, we want to always use double precision when creating a complex wave field."
                )

            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)

            if len(u.shape) == 2:  # [H, W]
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:  # [1, H, W]
                self.u = self.u.unsqueeze(0)

            self.res = self.u.shape[-2:]

        else:
            # Initialize a zero complex wave field
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res

        # Other paramters
        assert wvln > 0.1 and wvln < 1, "wvln unit should be [um]."
        self.wvln = wvln  # wvln, store in [um]
        self.k = 2 * torch.pi / (self.wvln * 1e-3)  # distance unit [mm]
        self.phy_size = phy_size  # physical size with padding, in [mm]
        self.valid_phy_size = (
            self.phy_size if valid_phy_size is None else valid_phy_size
        )  # physical size without padding, in [mm]

        assert phy_size[0] / self.res[0] == phy_size[1] / self.res[1], (
            "Wrong pixel size."
        )
        self.ps = phy_size[0] / self.res[0]  # pixel size, float value

        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(
            self.x, z
        )  # Maybe keeping z as a float tensor is better

    def load_img(self, img):
        """Load an image and use its pixel values as the amplitude of the complex wave field.
        The phase is initialized to zero everywhere.

        Args:
            img (torch.Tensor]): Input image with shape [H, W] or [B, C, H, W]. Data range is [0, 1].
        """
        assert img.dtype == torch.float32, "Image must be float32."

        amp = torch.sqrt(img)
        phi = torch.zeros_like(amp)

        self.u = amp + 1j * phi
        self.res = self.u.shape[-2:]
        return self

    def load(self, data_path):
        if data_path.endswith(".pkl"):
            self.load_pkl(data_path)
        else:
            raise Exception("Unimplemented file format.")

    def load_pkl(self, data_path):
        """Load data from pickle file."""
        with open(data_path, "rb") as tf:
            wave_data = pickle.load(tf)
            tf.close()

        amp = wave_data["amp"]
        phi = wave_data["phi"]
        self.u = amp * torch.exp(1j * phi)
        self.x = wave_data["x"]
        self.y = wave_data["y"]
        self.wvln = wave_data["wvln"]
        self.phy_size = wave_data["phy_size"]
        self.valid_phy_size = wave_data["valid_phy_size"]
        self.res = self.x.shape

    def save(self, save_path="./wavefield.pkl"):
        """Save the complex wave field to a pickle file."""
        self.save_data(save_path)

    def save_data(self, save_path="./wavefield.pkl"):
        """Save the complex wave field to a pickle file."""
        # Save data
        data = {
            "amp": self.u.cpu().abs(),
            "phi": torch.angle(self.u.cpu()),
            "x": self.x.cpu(),
            "y": self.y.cpu(),
            "wvln": self.wvln,
            "phy_size": self.phy_size,
            "valid_phy_size": self.valid_phy_size,
        }

        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)
            tf.close()

        # Save intensity, amplitude, and phase images
        u = self.u.cpu()
        save_image(u.abs() ** 2, f"{save_path[:-4]}_intensity.png", normalize=True)
        save_image(u.abs(), f"{save_path[:-4]}_amp.png", normalize=True)
        save_image(u.angle(), f"{save_path[:-4]}_phase.png", normalize=True)

    # =============================================
    # Operation
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
        wvln_mm = self.wvln * 1e-3
        valid_phy_size = self.valid_phy_size

        # Determine which propagation method to use
        if prop_dist < DELTA:
            # Zero distance: do nothing
            pass

        elif prop_dist < wvln_mm:
            # Sub-wavelength distance: full wave method
            raise Exception("Full wave method is not implemented.")

        else:
            # Other distances: Angular Spectrum Method
            prop_dist_min = Nyquist_zmin(
                wvln=self.wvln, ps=self.ps, max_side_dist=self.phy_size[0]
            )
            if np.abs(prop_dist) < prop_dist_min:
                print(
                    f"Minium required propagation distance is {prop_dist_min} mm, but propagation is still performed."
                )
            self.u = AngularSpectrumMethod(
                self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n
            )

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

    def gen_xy_grid(self):
        """Generate the x and y grid."""
        ps = self.ps
        x, y = torch.meshgrid(
            torch.linspace(
                -0.5 * self.phy_size[1] + 0.5 * ps,
                0.5 * self.phy_size[1] - 0.5 * ps,
                self.res[0],
            ),
            torch.linspace(
                0.5 * self.phy_size[0] - 0.5 * ps,
                -0.5 * self.phy_size[0] + 0.5 * ps,
                self.res[1],
            ),
            indexing="xy",
        )
        return x, y

    def gen_freq_grid(self):
        """Generate the frequency grid."""
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy

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
        """Pad the input field by (Hpad, Hpad, Wpad, Wpad). This step will also expand physical size of the field."""
        self.u = F.pad(self.u, (Hpad, Hpad, Wpad, Wpad), mode="constant", value=0)

        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [
            self.phy_size[0] * self.res[0] / Horg,
            self.phy_size[1] * self.res[1] / Worg,
        ]
        self.x, self.y = self.gen_xy_grid()
        z = self.z[0, 0]
        self.z = F.pad(self.z, (Hpad, Hpad, Wpad, Wpad), mode="constant", value=z)

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

    Reference:
        [1] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py#L293
        [2] https://blog.csdn.net/zhenpixiaoyang/article/details/111569495

    Args:
        u (tesor): complex field, shape [H, W] or [B, 1, H, W]
        z (float): propagation distance in [mm]
        wvln (float): wavelength in [um]
        ps (float): pixel size in [mm]
        n (float): refractive index
        padding (bool): padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]
    """
    assert wvln > 0.1 and wvln < 10, "wvln unit should be [um]."
    wvln_mm = wvln / n * 1e-3  # [um] to [mm]
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
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5 / ps, 0.5 / ps, Wimg, device=u.device),
        torch.linspace(0.5 / ps, -0.5 / ps, Himg, device=u.device),
        indexing="xy",
    )
    square_root = torch.sqrt(1 - wvln_mm**2 * (fx**2 + fy**2))
    H = torch.exp(1j * k * z * square_root)
    H = ifftshift(H)

    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifft2(fft2(u) * H)

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    del fx, fy
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

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.1
        [2] https://qiweb.tudelft.nl/aoi/wavefielddiffraction/wavefielddiffraction.html
        [3] https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m

    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        z (float): propagation distance
        wvln (float): wavelength in [um]
        ps (float): pixel size
        n (float): refractive index
        padding (bool): padding or not
        TF (bool): transfer function or impulse response
    """
    # Padding
    if padding:
        try:
            _, _, Worg, Horg = u.shape
        except:
            Horg, Worg = u.shape
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        _, _, Wimg, Himg = u.shape

    # Wave field parameters in medium
    assert wvln > 0.1 and wvln < 10, "wvln should be in [um]."
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

    Reference:
        [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Eq (9).
        [2] https://www.mathworks.com/matlabcentral/fileexchange/75049-complete-rayleigh-sommerfeld-model-version-2

    Args:
        u1: complex amplitude of input field, shape [H1, W1]
        x1: physical coordinate of input field, unit [mm], shape [H1, W1]
        y1: physical coordinate of input field, unit [mm], shape [H1, W1]
        z: propagation distance, unit [mm]
        wvln: wavelength, unit [um]
        x2: physical coordinate of output field, unit [mm], shape [H2, W2]
        y2: physical coordinate of output field, unit [mm], shape [H2, W2]
        n: refractive index
        memory_saving: memory saving

    Returns:
        u2: complex amplitude of output field, shape [H2, W2]
    """
    # Parameters
    assert wvln > 0.1 and wvln < 10, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3  # [um] to [mm]
    k = n * 2 * torch.pi / wvln_mm  # wave number [mm]-1
    if x2 is None:
        x2 = x1.clone()
    if y2 is None:
        y2 = y1.clone()

    # Nyquist sampling criterion
    max_side_dist = max(abs(x1.max() - x2.min()), abs(x2.max() - x1.min()))
    ps = (x1.max() - x1.min()) / x1.shape[-1]
    zmin = Nyquist_zmin(
        wvln=wvln, ps=ps.item(), max_side_dist=max_side_dist.item(), n=n
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


def Nyquist_zmin(wvln, ps, max_side_dist, n=1.0):
    """Nyquist sampling condition for Rayleigh Sommerfeld diffraction.

    Reference:
        [1] Is the Rayleigh-Sommerfeld diffraction always an exact reference for high speed diffraction algorithms? Optics Express 2017.

    Args:
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        max_side_dist: maximum side distance between input and output field in [mm]
        n: refractive index

    Returns:
        zmin: minimum propagation distance in [mm] required by Nyquist sampling criterion
    """
    wvln_mm = wvln * 1e-3
    zmin = np.sqrt((4 * ps**2 * n**2 / wvln_mm**2 - 1)) * max_side_dist
    return round(zmin, 3)

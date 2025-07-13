# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Wave optics utilities."""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deeplens.optics.basics import DEFAULT_WAVE
from deeplens.optics.wave import ComplexWave


# ==================================
# Commonly used wave fields
# ==================================
def plane_wave_field(
    phy_size=(2, 2),
    res=(1000, 1000),
    wvln=0.589,
    z=0.0,
    valid_r=None,
):
    """Create a planar wave field on x0y plane.

    Args:
        phy_size (tuple): Physical size of the field. [mm].
        res (tuple): Resolution.
        wvln (float): Wavelength. [um].
        z (float): Field z position. [mm].
        valid_r (float): Valid circle radius. [mm].

    Returns:
        field (ComplexWave): Complex field.
    """
    assert wvln > 0.1 and wvln < 1.0, "wvln should be in [um]."

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
    return ComplexWave(u=u, phy_size=phy_size, wvln=wvln, res=res, z=z)


def point_source_field(
    point=(0, 0, -1000.0),
    phy_size=(2, 2),
    res=(1000, 1000),
    z=0.0,
    wvln=0.589,
    valid_r=None,
):
    """Create a spherical wave field on x0y plane originating from a point source.

    Args:
        point (tuple): Point source position in object space. [mm]. Defaults to (0, 0, -1000.0).
        phy_size (tuple): Valid plane on x0y plane. [mm]. Defaults to (2, 2).
        res (tuple): Valid plane resoltution. Defaults to (1000, 1000).
        z (float): Field z position. [mm]. Defaults to 0.0.
        wvln (float): Wavelength. [um]. Defaults to 0.589.
        valid_r (float): Valid circle radius. [mm]. Defaults to None.

    Returns:
        field (ComplexWave): Complex field on x0y plane.
    """
    assert wvln > 0.1 and wvln < 1.0, "wvln should be in [um]."
    k = 2 * torch.pi / (wvln * 1e-3)  # k in [mm^-1]

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

    # Apply valid circle if provided
    if valid_r is not None:
        mask = (x - point[0]) ** 2 + (y - point[1]) ** 2 < valid_r**2
        u = u * mask

    # Create wave field
    return ComplexWave(u=u, wvln=wvln, phy_size=phy_size, res=res, z=z)


def square_field(n, W, w, x0=0, y0=0, theta=0):
    """Code copied from yidan's rect function.

    Args:
        n (int): Resolution.
        W (float): Physical size.
        w (float): Rect width.
        x0 (float): Rect center x.
        y0 (float): Rect center y.
        theta (float): Rotation angle.

    Returns:
        u (torch.Tensor): Rect field.
    """
    raise Exception("This function is deprecated.")
    x, y = torch.meshgrid(
        torch.linspace(-W / 2, W / 2, n),
        torch.linspace(-W / 2, W / 2, n),
        indexing="xy",
    )

    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    u = ((xr - x0).abs() <= w) * ((yr - y0).abs() < w)
    u = u.float()

    return u


def circle_field(phy_size, circle_radius, W, H, wvln=DEFAULT_WAVE, center=[0, 0]):
    """Generate a circle plane field.

    Args:
        phy_size (list): Physical size of the field. [mm].
        circle_radius (float): Circle radius. [mm].
        W (int): Resolution in x direction.
        H (int): Resolution in y direction.
        center (list): Circle center. [mm]. Defaults to [0, 0].

    Returns:
        field (ComplexWave): Complex field.
    """
    raise Exception("This function is deprecated.")
    x, y = torch.meshgrid(
        torch.linspace(-phy_size[0] / 2, phy_size[0] / 2, H),
        torch.linspace(-phy_size[1] / 2, phy_size[1] / 2, W),
        indexing="xy",
    )
    u = torch.zeros((H, W)) + 0j
    circle_idx = (x - center[0]) ** 2 + (y - center[1]) ** 2 < circle_radius**2
    u[circle_idx] = 1

    return ComplexWave(u=u, phy_size=phy_size, wvln=wvln)


def sphere_wave(
    source,
    x_range=[-1, 1],
    y_range=[-1, 1],
    z_pos=0.0,
    wvln=0.589,
    res=[1000, 1000],
    converge=True,
):
    """Generate a sphere wave field.

    Args:
        source (list): Source position. [mm].
        x_range (list): x range. [mm].
        y_range (list): y range. [mm].
        res (list): Resolution.

    Returns:
        field (ComplexWave): Complex field.
    """
    raise Exception("This function is deprecated.")
    x, y = torch.meshgrid(
        torch.linspace(x_range[0], x_range[1], res[1], dtype=torch.float64),
        torch.linspace(y_range[1], y_range[0], res[0], dtype=torch.float64),
        indexing="xy",
    )
    z = torch.full_like(x, z_pos)
    r = torch.sqrt((x - source[0]) ** 2 + (y - source[1]) ** 2 + (z - source[2]) ** 2)
    if converge:
        u = torch.exp(-1j * 2 * np.pi / (wvln * 1e-3) * r) * r.max() / r
    else:
        u = torch.exp(1j * 2 * np.pi / (wvln * 1e-3) * r) * r.max() / r

    xc, yc = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
    mask = ((x - xc) ** 2 + (y - yc) ** 2) > (0.5 * (x_range[1] - x_range[0])) ** 2
    u[mask] = 0 + 0j
    return ComplexWave(
        u=u,
        phy_size=[x_range[1] - x_range[0], y_range[1] - y_range[0]],
        z=z_pos,
        res=res,
        wvln=wvln,
    )


# ==================================
# Image batch to wave field
# ==================================
def img2field(img=None, wvln=0.589, phy_size=[1, 1], padding=False):
    """Convert a monochrome image to a complex field.

        phase = 0, amplitutu = image

        Reference: https://iust-projects.ir/post/cv02/
        (This reference is used to distill amp and phi information from an image)

    Args:
        img (np.array): Image. Defaults to None.
        wvln (float): wvln. Defaults to 0.589.
        phy_size (list): Physical size of the field. Defaults to [1,1].
        padding (bool): Pad the field to be a power of 2. Defaults to False.
        device (str): 'cuda' or 'cpu'. Defaults to 'cuda'.

    Returns:
        field (ComplexWave): Complex field.
    """
    if img is None:
        img = cv.imread("dataset/lena.png", cv.IMREAD_GRAYSCALE)

    if len(img.shape) > 2:
        raise Exception("Only monochrome images supported now.")

    H, W = img.shape
    amp = torch.sqrt(torch.from_numpy(img / 255.0))
    phi = torch.zeros_like(amp)
    u = amp + 1j * phi

    if padding:
        u = F.pad(u, (H // 4, H // 4, W // 4, W // 4), mode="constant", value=0)

    res = u.shape
    field = ComplexWave(u=u, phy_size=phy_size, res=res, wvln=wvln)

    return field


def batch2field(
    img, phy_size, z=0, wvln=0.589, padding=False, phase="zero", device="cpu"
):
    """Convert a batch of images to a complex field.

    Args:
        img (np.array): Image. Defaults to None.
        wvln (float): wvln. Defaults to 0.589.
        phy_size (list): Physical size of the field. Defaults to [1,1].
        padding (bool): Pad the field to be a power of 2. Defaults to False.
        phase (str): Phase initialization method. Defaults to 'zero'.
        device (str): 'cuda' or 'cpu'. Defaults to 'cuda'.

    Returns:
        field (ComplexWave): Complex field.
    """
    B, C, H, W = img.shape

    # Amplitude initialization
    if torch.is_tensor(img):
        amp = torch.sqrt(img)
    else:
        amp = torch.sqrt(torch.from_numpy(img))

    # Phase initialization
    if phase == "zero":
        phi = torch.zeros_like(amp)
    elif phase == "amp":
        phi = amp.clone() * 2 * np.pi
    elif phase == "random":
        phi = torch.rand_like(amp)
    else:
        phi = amp.clone() * 2 * np.pi

    # Complex wave
    u = amp * torch.exp(1j * phi)
    valid_phy_size = phy_size

    # Padding
    if padding:
        _, _, Worg, Horg = u.shape
        Wpad, Hpad = int(Worg * 2), int(Horg * 2)
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
        # u = F.pad(u, (H//2,H//2,W//2,W//2), mode='constant', value=0.)
        phy_size = [Wimg / Worg * phy_size[0], Himg / Horg * phy_size[0]]

    res = u.shape
    field = ComplexWave(
        u=u, phy_size=phy_size, valid_phy_size=valid_phy_size, res=res, wvln=wvln
    ).to(device)
    field.z += z

    return field


def field2img(amp, phase):
    """Convert a complex field to a RGB image.

        Reference: https://iust-projects.ir/post/cv02/
    Args:
        amp (_type_): _description_
        phase (_type_): _description_
    """
    img_comb = np.multiply(amp, np.exp(1j * phase))
    img = np.real(np.fft.ifft2(img_comb))  # drop imagniary as they are around 1e-14

    plt.imshow(np.abs(img), cmap="gray")

    return img

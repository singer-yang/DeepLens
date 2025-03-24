"""PSF-related functions."""

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F


# ================================================
# PSF convolution
# ================================================
def render_psf(img, psf):
    """Render rgb image batch with rgb PSF.

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]

    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    # Padding
    _, ks, ks = psf.shape
    padding = int(ks / 2)
    psf = torch.flip(psf, [1, 2])  # flip the PSF because F.conv2d use cross-correlation
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]
    img_pad = F.pad(img, (padding, padding, padding, padding), mode="reflect")

    # Convolution
    img_render = F.conv2d(img_pad, psf, groups=img.shape[1], padding=0, bias=None)
    return img_render


def render_psf_map(img, psf_map):
    """Render a rgb image batch with PSF map using patch convolution.

    Args:
        img (torch.Tensor): [B, 3, H, W]
        psf_map (torch.Tensor): [grid_h, grid_w, 3, ks, ks]

    Returns:
        render_img (torch.Tensor): [B, C, H, W]
    """
    # Patch convolution
    grid_h, grid_w, _, ks, ks = psf_map.shape
    _, _, Himg, Wimg = img.shape
    pad = int(ks / 2)
    img_pad = F.pad(img, (pad, pad, pad, pad), mode="reflect")

    # Render image patch by patch
    render_img = torch.zeros_like(img)
    for i in range(grid_h):
        for j in range(grid_w):
            psf = psf_map[i, j]  # shape [C, ks, ks]
            psf = torch.flip(psf, [1, 2]).unsqueeze(1)  # shape [C, 1, ks, ks]

            h_low, w_low = int(i / grid_h * Himg), int(j / grid_w * Wimg)
            h_high, w_high = int((i + 1) / grid_h * Himg), int((j + 1) / grid_w * Wimg)

            # Consider overlap to avoid boundary artifacts
            img_pad_patch = img_pad[
                :, :, h_low : h_high + 2 * pad, w_low : w_high + 2 * pad
            ]
            render_patch = F.conv2d(
                img_pad_patch, psf, groups=img.shape[1], padding="valid", bias=None
            )
            render_img[:, :, h_low:h_high, w_low:w_high] = render_patch

    return render_img


def local_psf_render(input, psf):
    """Render an image with pixel-wise PSF. Use the different PSF kernel for different pixels (folding approach).

        Application example: Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (B, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, 3, ks, ks)

    Returns:
        output (Tensor): Rendered image (B, C, H, W)
    """
    # Folding for convolution
    B, Cimg, Himg, Wimg = input.shape
    Hpsf, Wpsf, Cpsf, Ks, Ks = psf.shape
    assert Cimg == Cpsf and Himg == Hpsf and Wimg == Wpsf, (
        "Input and PSF shape mismatch"
    )
    pad = int((Ks - 1) / 2)

    # 1. Pad the input with replicated values
    inp_pad = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")

    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = psf.reshape(Himg * Wimg, 3, Ks, Ks)
    kernels_flip = torch.flip(kernels, [-2, -1])

    # 3. Unfold input
    inp_unf = F.unfold(inp_pad, (Ks, Ks))  # [B, C*Ks*Ks, H*W]

    # 4. Reshape for efficient computation
    inp_unf = inp_unf.view(B, Cimg, Ks * Ks, Himg * Wimg)  # [B, C, Ks*Ks, H*W]
    kernels_flip = kernels_flip.view(Himg * Wimg, 3, Ks * Ks)  # [H*W, 3, Ks*Ks]

    # 5. Use einsum for efficient batch-wise multiplication and summation
    # This computes the dot product between each unfolded patch and its corresponding kernel
    # for each batch and channel
    y = torch.zeros(B, 3, Himg * Wimg, device=input.device)

    for b in range(B):  # Still need one loop for batch, but channels are vectorized
        # einsum: 'ckp,pck->cp' means:
        # c: channel dimension
        # k: kernel elements (Ks*Ks)
        # p: pixel positions (H*W)
        # Multiply corresponding elements and sum over k
        y[b] = torch.einsum("ckp,pck->cp", inp_unf[b], kernels_flip)

    # 6. Fold and return
    img = F.fold(y, (Himg, Wimg), (1, 1))
    return img


def local_psf_render_high_res(input, psf, patch_num=[4, 4], overlap=0.2):
    """Render an image with pixel-wise PSF using patch-wise rendering. Overlapping windows are used to avoid boundary artifacts.

    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, 3, ks, ks)
        patch_num (list): Number of patches in each dimension. Defaults to [4, 4].
        overlap (float): Fraction of overlap between adjacent patches (0-1). Defaults to 0.2.

    Returns:
        Tensor: Rendered image with same shape as input.
    """
    B, Cimg, Himg, Wimg = input.shape
    Hpsf, Wpsf, Cpsf, Ks, Ks = psf.shape
    assert Cimg == Cpsf and Himg == Hpsf and Wimg == Wpsf, (
        "Input and PSF shape mismatch"
    )

    # Calculate base patch size
    base_patch_h = Himg // patch_num[0]
    base_patch_w = Wimg // patch_num[1]

    # Calculate overlap in pixels
    overlap_h = int(base_patch_h * overlap)
    overlap_w = int(base_patch_w * overlap)

    # Initialize output and weight accumulation tensors
    img_render = torch.zeros_like(input)
    weight_accumulation = torch.zeros((B, 1, Himg, Wimg), device=input.device)

    # Create weight mask for blending (higher weight in center, lower at edges)
    def create_weight_mask(h, w):
        y = torch.linspace(0, 1, h, device=input.device)
        x = torch.linspace(0, 1, w, device=input.device)

        # Create 2D weight grid (higher in center, lower at edges)
        y = torch.min(y, 1 - y) * 2  # Transform to [0->1->0]
        x = torch.min(x, 1 - x) * 2  # Transform to [0->1->0]

        # Create 2D weight grid
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        # Combine weights (multiply or min for smoother transition)
        weights = torch.min(y_grid, x_grid).unsqueeze(0).unsqueeze(0)

        # Apply non-linearity for sharper transition
        weights = weights**2

        return weights

    # Process each patch with overlap
    for pi in range(patch_num[0]):
        for pj in range(patch_num[1]):
            # Calculate patch boundaries with overlap
            low_i = max(0, pi * base_patch_h - overlap_h)
            up_i = min(Himg, (pi + 1) * base_patch_h + overlap_h)
            low_j = max(0, pj * base_patch_w - overlap_w)
            up_j = min(Wimg, (pj + 1) * base_patch_w + overlap_w)

            # Extract patches
            img_patch = input[:, :, low_i:up_i, low_j:up_j]
            psf_patch = psf[low_i:up_i, low_j:up_j, :, :, :]

            # Process patch
            rendered_patch = local_psf_render(img_patch, psf_patch)

            # Create weight mask for this patch
            patch_h, patch_w = up_i - low_i, up_j - low_j
            weight_mask = create_weight_mask(patch_h, patch_w)

            # Accumulate weighted result
            img_render[:, :, low_i:up_i, low_j:up_j] += rendered_patch * weight_mask
            weight_accumulation[:, :, low_i:up_i, low_j:up_j] += weight_mask

    # Normalize by accumulated weights to blend patches
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    img_render = img_render / (weight_accumulation + epsilon)

    return img_render


# ================================================
# PSF map operations
# ================================================
def crop_psf_map(psf_map, grid, ks_crop, psf_center=None):
    """Crop the center part of each PSF patch.

    Args:
        psf_map (torch.Tensor): [C, grid*ks, grid*ks]
        grid (int): grid number
        ks_crop (int): cropped PSF kernel size
        psf_center (torch.Tensor): (grid, grid, 2) center of the PSF patch

    Returns:
        psf_map_crop (torch.Tensor): [C, grid*ks_crop, grid*ks_crop]
    """
    if len(psf_map.shape) == 4:
        psf_map = psf_map.squeeze(0)
    C, H, W = psf_map.shape
    assert H % grid == 0 and W % grid == 0, "PSF map size should be divisible by grid"
    ks = int(H / grid)
    assert ks % 2 == 1, "PSF kernel size should be odd"

    psf_map_crop = torch.zeros((C, grid * ks_crop, grid * ks_crop)).to(psf_map.device)
    for i in range(grid):
        for j in range(grid):
            psf = psf_map[:, i * ks : (i + 1) * ks, j * ks : (j + 1) * ks]

            # Without re-center
            if psf_center is None:
                psf_crop = psf[
                    :,
                    int((ks - ks_crop) / 2) : int((ks + ks_crop) / 2),
                    int((ks - ks_crop) / 2) : int((ks + ks_crop) / 2),
                ]
            else:
                raise Exception("Not tested")
                psf_crop = psf[
                    :,
                    psf_center[0] - int((ks_crop - 1) / 2) : psf_center[0]
                    + int((ks_crop + 1) / 2),
                    psf_center[1] - int((ks_crop - 1) / 2) : psf_center[1]
                    + int((ks_crop + 1) / 2),
                ]

            # Normalize cropped PSF
            psf_crop[0, :, :] = psf_crop[0, :, :] / torch.sum(psf_crop[0, :, :])
            psf_crop[1, :, :] = psf_crop[1, :, :] / torch.sum(psf_crop[1, :, :])
            psf_crop[2, :, :] = psf_crop[2, :, :] / torch.sum(psf_crop[2, :, :])

            # Put cropped PSF into the map
            psf_map_crop[
                :, i * ks_crop : (i + 1) * ks_crop, j * ks_crop : (j + 1) * ks_crop
            ] = psf_crop

    return psf_map_crop


def interp_psf_map(psf_map, grid_old, grid_new):
    """Interpolate the PSF map from [C, grid_old*ks, grid_old*ks] to [C, grid_new*ks, grid_new*ks]. Usecase: I want to interpolate the PSF map from 10x10 grid to 20x20 grid.

    Args:
        psf_map (torch.Tensor): [C, grid_old*ks, grid_old*ks]
        grid_old (int): old grid number
        grid_new (int): new grid number

    Returns:
        psf_map_interp (torch.Tensor): [C, grid_new*ks, grid_new*ks]
    """
    if len(psf_map.shape) == 3:
        # [C, grid_old*ks, grid_old*ks]
        C, H, W = psf_map.shape
        assert H % grid_old == 0 and W % grid_old == 0, (
            "PSF map size should be divisible by grid"
        )
        ks = int(H / grid_old)
        assert ks % 2 == 1, "PSF kernel size should be odd"

        # Reshape from [C, grid*ks, grid*ks] to [grid_old, grid_old, C, ks, ks]
        psf_map_interp = psf_map.reshape(C, grid_old, ks, grid_old, ks).permute(
            1, 3, 0, 2, 4
        )  # .reshape(grid_old, grid_old, C, ks, ks)
    elif len(psf_map.shape) == 5:
        # [grid_old, grid_old, C, ks, ks]
        grid_old, grid_old, C, ks, ks = psf_map.shape
        psf_map_interp = psf_map
    else:
        raise ValueError(
            "PSF map should be [C, grid_old*ks, grid_old*ks] or [grid_old, grid_old, C, ks, ks]"
        )

    # Reshape from [grid_old, grid_old, C, ks, ks] to [ks*ks, C, grid_old, grid_old]
    psf_map_interp = psf_map_interp.permute(3, 4, 2, 0, 1).reshape(
        ks * ks, C, grid_old, grid_old
    )

    # Interpolate from [ks*ks, C, grid_old, grid_old] to [ks*ks, C, grid_new, grid_new]
    psf_map_interp = F.interpolate(
        psf_map_interp, size=(grid_new, grid_new), mode="bilinear", align_corners=True
    )

    # Reshape from [ks*ks, C, grid_new, grid_new] to [C, grid_new*ks, grid_new*ks]
    psf_map_interp = (
        psf_map_interp.reshape(ks, ks, C, grid_new, grid_new)
        .permute(2, 3, 0, 4, 1)
        .reshape(C, grid_new * ks, grid_new * ks)
    )

    return psf_map_interp


def read_psf_map(filename, grid=10):
    """Read PSF map from a PSF map image.

    Args:
        filename (str): path to the PSF map image
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    psf_map = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
    psf_map = torch.tensor(psf_map).permute(2, 0, 1).float() / 255.0
    psf_ks = psf_map.shape[-1] // grid
    for i in range(grid):
        for j in range(grid):
            psf_map[
                0, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[0, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )
            psf_map[
                1, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[1, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )
            psf_map[
                2, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[2, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )

    return psf_map


# ================================================
# Inverse PSF calculation
# ================================================
def solve_psf(img_org, img_render, ks=51, eps=1e-6):
    """Solve PSF, where img_render = img_org * psf.

    Args:
        img_org (torch.Tensor): The object image tensor of shape [1, 3, H, W].
        img_render (torch.Tensor): The simulated/observed image tensor of shape [1, 3, H, W].
        eps (float): A small epsilon value to prevent division by zero in frequency domain.

    Returns:
        psf (torch.Tensor): The PSF tensor of shape [3, ks, ks].
    """
    # Move to frequency domain
    F_org = torch.fft.fftn(img_org, dim=[2, 3])
    F_render = torch.fft.fftn(img_render, dim=[2, 3])

    # Solve for F_psf in frequency domain
    F_psf = F_render / (F_org + eps)

    # Inverse FFT to get PSF in spatial domain
    # Here, we take the real part assuming the PSF should be real-valued
    psf = torch.fft.ifftn(F_psf, dim=[2, 3]).real
    psf = torch.fft.fftshift(psf, dim=[2, 3])

    # Crop to get PSF size [3, 51, 51]
    _, _, H, W = psf.shape
    start_h = (H - ks) // 2
    start_w = (W - ks) // 2
    psf = psf[0, :, start_h : start_h + ks, start_w : start_w + ks]

    # Normalize PSF to sum to 1
    psf = psf / torch.sum(psf, dim=[1, 2], keepdim=True)

    return psf


def solve_psf_map(img_org, img_render, ks=51, grid=10):
    """Solve PSF map by inverse convolution.

    Args:
        img_org (torch.Tensor): [B, 3, H, W]
        img_render (torch.Tensor): [B, 3, H, W]
        ks (int): PSF kernel size
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    assert img_org.shape[-1] == img_org.shape[-2], "Image should be square"
    assert (img_org.shape[-1] % grid == 0) and (img_org.shape[-2] % grid == 0), (
        "Image size should be divisible by grid"
    )
    patch_size = int(img_org.shape[-1] / grid)
    psf_map = torch.zeros((3, grid * ks, grid * ks)).to(img_org.device)

    for i in range(grid):
        for j in range(grid):
            img_org_patch = img_org[
                :,
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            img_render_patch = img_render[
                :,
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            psf_patch = solve_psf(img_org_patch, img_render_patch, ks=ks)

            psf_map[:, i * ks : (i + 1) * ks, j * ks : (j + 1) * ks] = psf_patch

    return psf_map

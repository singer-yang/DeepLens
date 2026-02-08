# Copyright 2026 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF-related functions.

PSF convolution functions:
    PSF for image patch simulation.
        - conv_psf(): a single PSF kernel for the whole patch, no spatial variation or defocus.
        - conv_psf_depth_interp(): depth-varying PSF for the whole patch, no spatial variation.
    
    PSF map.
        - conv_psf_map(): a PSF map for the whole image, spatial varying across different image patches, no spatial variation within the patch, no defocus.
        - conv_psf_map_depth_interp(): depth-varying PSF map for the whole image, spatial varying across different image patches, no spatial variation within the patch.
    
    Per-pixel PSF. 
        - conv_psf_pixel(): each pixel has a unique PSF, spatial variance and defocus.

Other functions:
    - crop_psf_map(): crop a PSF map to a smaller size.
    - interp_psf_map(): interpolate a PSF map to a different grid size.
    - read_psf_map(): read a PSF map from a file.
    - rotate_psf(): rotate a PSF kernel.
    - solve_psf(): solve a PSF kernel from a given image and rendered image.
    - solve_psf_map(): solve a PSF map from a given image and rendered image.
"""

import cv2 as cv
import torch
import torch.nn.functional as F

from deeplens.basics import DELTA, PSF_KS


# ================================================
# PSF convolution for image simulation
# ================================================

def conv_psf(img, psf):
    """Convolve an image batch with a PSF.

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]. ks can be odd or even.

    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    B, C, H, W = img.shape
    C_psf, ks, _ = psf.shape
    assert C_psf == C, f"psf channels ({C_psf}) must match image channels ({C})."

    # Flip the PSF because F.conv2d use cross-correlation
    psf = torch.flip(psf, [1, 2])
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]

    # Padding
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Convolution
    img_render = F.conv2d(img_pad, psf, groups=C)
    return img_render

def conv_psf_map(img, psf_map):
    """Convolve an image batch with a PSF map.

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf_map (torch.Tensor): [grid_h, grid_w, C, ks, ks]

    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    B, C, H, W = img.shape
    grid_h, grid_w, C_psf, ks, _ = psf_map.shape
    assert C_psf == C, f"PSF map channels ({C_psf}) must match image channels ({C})."
    
    # Padding
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Pre-flip entire PSF map once (instead of flipping each PSF inside the loop)
    psf_map_flipped = torch.flip(psf_map, dims=(-2, -1))

    # Render image patch by patch
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w

            # PSF, [C, 1, ks, ks]
            psf = psf_map_flipped[i, j].unsqueeze(1)

            # Consider overlap to avoid boundary artifacts
            img_pad_patch = img_pad[
                :,
                :,
                h_low : h_high + pad_h_left + pad_h_right,
                w_low : w_high + pad_w_left + pad_w_right,
            ]

            # Convolution, [B, C, h_high-h_low, w_high-w_low]
            render_patch = F.conv2d(img_pad_patch, psf, groups=C)  
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render


def conv_psf_map_depth_interp(img, depth, psf_map, psf_depths, interp_mode="depth"):
    """Convolve an image with a PSF map. Within each image patch, do interpolation with a depth map.

    Args:
        img: (B, 3, H, W), [0, 1]
        depth: (B, 1, H, W), (-inf, 0)
        psf_map: (grid_h, grid_w, num_depth, 3, ks, ks)
        psf_depths: (num_depth). (-inf, 0). Used to interpolate psf_map.
        interp_mode: "depth" or "disparity". If "disparity", weights are calculated based on disparity (1/depth).
    
    Returns:
        img_render: (B, 3, H, W), [0, 1]
    """
    assert interp_mode in ["depth", "disparity"], f"interp_mode must be 'depth' or 'disparity', got {interp_mode}"
    assert depth.min() < 0 and depth.max() < 0, f"depth must be negative, got {depth.min()} and {depth.max()}"
    assert psf_depths.min() < 0 and psf_depths.max() < 0, f"psf_depths must be negative, got {psf_depths.min()} and {psf_depths.max()}"

    B, C, H, W = img.shape
    grid_h, grid_w, num_depths, C_psf, ks, _ = psf_map.shape
    assert C_psf == C, f"PSF map channels ({C_psf}) must match image channels ({C})."

    # Pad the full image once to avoid boundary artifacts at patch seams.
    # Without this, each patch would be padded independently (reflecting within
    # its own boundary), producing visible seams at grid boundaries.
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Pre-flip entire PSF map once: [grid_h, grid_w, num_depths, C, ks, ks]
    psf_map_flipped = torch.flip(psf_map, dims=(-2, -1))

    # Pre-compute depth interpolation weights (shared across all patches)
    depth_flat = depth.flatten(1)  # [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0] + DELTA, psf_depths[-1] - DELTA)
    indices = torch.searchsorted(psf_depths, depth_flat, right=True)  # [B, H*W]
    indices = indices.clamp(1, num_depths - 1)
    idx0 = indices - 1
    idx1 = indices

    d0 = psf_depths[idx0]  # [B, H*W]
    d1 = psf_depths[idx1]

    if interp_mode == "depth":
        denom = d1 - d0
        denom[denom == 0] = 1e-6
        w1 = (depth_flat - d0) / denom
    else:
        disp_flat = 1.0 / depth_flat
        disp0 = 1.0 / d0
        disp1 = 1.0 / d1
        denom = disp1 - disp0
        denom[denom == 0] = 1e-6
        w1 = (disp_flat - disp0) / denom

    w0 = 1 - w1

    # Reshape weight indices to spatial layout for patch extraction
    idx0_spatial = idx0.view(B, H, W)
    idx1_spatial = idx1.view(B, H, W)
    w0_spatial = w0.view(B, H, W)
    w1_spatial = w1.view(B, H, W)

    # Render image patch by patch
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h
        patch_h = h_high - h_low

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w
            patch_w = w_high - w_low

            # Extract overlapping patch from pre-padded image (no per-patch padding needed)
            img_pad_patch = img_pad[
                :, :,
                h_low : h_high + pad_h_left + pad_h_right,
                w_low : w_high + pad_w_left + pad_w_right,
            ]

            # Expand patch for all depths: [B, C, patch_h+pad, patch_w+pad] -> [B, num_depths*C, ...]
            img_patch_expanded = img_pad_patch.unsqueeze(1).expand(B, num_depths, C, -1, -1).reshape(
                B, num_depths * C, img_pad_patch.shape[2], img_pad_patch.shape[3]
            )

            # PSF kernels for this grid cell: [num_depths*C, 1, ks, ks]
            psf_stacked = psf_map_flipped[i, j].reshape(num_depths * C, 1, ks, ks)

            # Grouped convolution -> [B, num_depths*C, patch_h, patch_w]
            patch_blur = F.conv2d(img_patch_expanded, psf_stacked, groups=num_depths * C)

            # Reshape to [num_depths, B, C, patch_h, patch_w]
            patch_blur = patch_blur.reshape(B, num_depths, C, patch_h, patch_w).permute(1, 0, 2, 3, 4)

            # Extract pre-computed weights for this patch
            patch_idx0 = idx0_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_idx1 = idx1_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_w0 = w0_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_w1 = w1_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)

            # Build per-depth weight tensor for this patch
            weights = torch.zeros(num_depths, B, patch_h * patch_w, device=img.device, dtype=img.dtype)
            weights.scatter_add_(0, patch_idx0.unsqueeze(0).long(), patch_w0.unsqueeze(0))
            weights.scatter_add_(0, patch_idx1.unsqueeze(0).long(), patch_w1.unsqueeze(0))
            weights = weights.view(num_depths, B, 1, patch_h, patch_w)

            # Apply depth-interpolation weights
            render_patch = torch.sum(patch_blur * weights, dim=0)
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render


def conv_psf_depth_interp(img, depth, psf_kernels, psf_depths, interp_mode="depth"):
    """Convolve an image batch with PSFs at multiple given depths, then do interpolation with a depth map.

    The differentiability of this function is not guaranteed.

    Args:
        img: (B, 3, H, W), [0, 1]
        depth: (B, 1, H, W), (-inf, 0)
        psf_kernels: (num_depth, 3, ks, ks)
        psf_depths: (num_depth). (-inf, 0). Used to interpolate psf_kernels.
        interp_mode: "depth" or "disparity". If "disparity", weights are calculated based on disparity (1/depth).

    Returns:
        img_blur: (B, 3, H, W), [0, 1]
    """
    assert interp_mode in ["depth", "disparity"], f"interp_mode must be 'depth' or 'disparity', got {interp_mode}"
    assert depth.min() < 0 and depth.max() < 0, f"depth must be negative, got {depth.min()} and {depth.max()}"
    assert psf_depths.min() < 0 and psf_depths.max() < 0, f"psf_depths must be negative, got {psf_depths.min()} and {psf_depths.max()}"
    
    # assert img.device != torch.device("cpu"), "Image must be on GPU"
    num_depths, _, ks, _ = psf_kernels.shape

    # =================================
    # PSF convolution for all depths
    # =================================
    B, C, H, W = img.shape
    
    # Prepare PSF kernel: [num_depths, C, ks, ks] -> [num_depths*C, 1, ks, ks]
    # Flip the PSF because F.conv2d uses cross-correlation
    psf_stacked = torch.flip(psf_kernels, [-2, -1]).reshape(num_depths * C, 1, ks, ks)

    # Pad before expand: pad [B, C, H, W] first (C channels), then expand to num_depths*C
    # This reduces padding work by a factor of num_depths
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_padded_small = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Expand padded img: [B, C, H+pad, W+pad] -> [B, num_depths*C, H+pad, W+pad]
    img_padded = img_padded_small.unsqueeze(1).expand(B, num_depths, C, -1, -1).reshape(B, num_depths * C, img_padded_small.shape[2], img_padded_small.shape[3])
    
    # Grouped convolution: each of the num_depths*C channels is convolved with its own kernel
    imgs_blur = F.conv2d(img_padded, psf_stacked, groups=num_depths * C)  # [B, num_depths*C, H, W]
    
    # Reshape to [num_depths, B, C, H, W]
    imgs_blur = imgs_blur.reshape(B, num_depths, C, H, W).permute(1, 0, 2, 3, 4)

    # =================================
    # Depth/Disparity interpolation
    # =================================
    B, _, H, W = depth.shape
    depth_flat = depth.flatten(1)  # shape [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0] + DELTA, psf_depths[-1] - DELTA)
    indices = torch.searchsorted(psf_depths, depth_flat, right=True)  # shape [B, H*W]
    indices = indices.clamp(1, num_depths - 1)
    idx0 = indices - 1
    idx1 = indices

    # Calculate weights for depth interpolation
    d0 = psf_depths[idx0]  # shape [B, H*W]
    d1 = psf_depths[idx1]
    
    if interp_mode == "depth":
        # Interpolate in depth space
        denom = d1 - d0
        denom[denom == 0] = 1e-6  # Avoid division by zero
        w1 = (depth_flat - d0) / denom  # shape [B, H*W]
    else:
        # Interpolate in disparity space (disparity = 1/depth)
        disp_flat = 1.0 / depth_flat
        disp0 = 1.0 / d0
        disp1 = 1.0 / d1
        denom = disp1 - disp0
        denom[denom == 0] = 1e-6  # Avoid division by zero
        w1 = (disp_flat - disp0) / denom  # shape [B, H*W]
    
    w0 = 1 - w1

    # Create a weight tensor
    weights = torch.zeros(num_depths, B, H * W, device=img.device, dtype=img.dtype)
    weights.scatter_add_(0, idx0.unsqueeze(0).long(), w0.unsqueeze(0))
    weights.scatter_add_(0, idx1.unsqueeze(0).long(), w1.unsqueeze(0))
    weights = weights.view(num_depths, B, 1, H, W)

    # Apply weights to the blurred images
    img_render = torch.sum(imgs_blur * weights, dim=0)
    return img_render

def conv_psf_pixel(img, psf):
    """Convolve an image batch with pixel-wise PSF.

    Use the different PSF kernel for different pixels (folding approach). Application example: Blurs image with dynamic Gaussian blur.

    Args:
        img (Tensor): The image to be blurred (B, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, C, ks, ks). ks can be odd or even.
    
    Returns:
        img_render (Tensor): Rendered image (B, C, H, W).
    """
    B, C, H, W = img.shape
    H_psf, W_psf, C_psf, ks, _ = psf.shape
    assert C == C_psf, ("Image and PSF channels mismatch.")
    assert H == H_psf and W == W_psf, ("Image and PSF size mismatch.")

    # Scattering for PSF convolution
    img = img.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
    kernels = psf.permute(2, 0, 1, 3, 4).unsqueeze(0)  # [1, C, H, W, ks, ks]
    y = img * kernels  # [B, C, H, W, ks, ks]

    # Fold the result, shape [B, C*ks*ks, H*W]
    y = y.permute(0, 1, 4, 5, 2, 3).reshape(B, C * ks * ks, H * W)

    # Output processing
    if ks % 2 == 0:
        pad_h_left  = (ks - 1) // 2
        pad_h_right = ks // 2
        pad_w_left  = (ks - 1) // 2
        pad_w_right = ks // 2
        img_render = F.fold(y, (H + pad_h_left + pad_h_right, W + pad_w_left + pad_w_right), (ks, ks), padding=0)
        img_render = img_render[:, :, pad_h_left:-pad_h_right, pad_w_left:-pad_w_right]
    else:
        pad = (ks - 1) // 2
        img_render = F.fold(y, (H, W), (ks, ks), padding=pad)
    
    return img_render

def conv_psf_pixel_high_res(img, psf, patch_num=(4, 4), expand=False):
    """Convolve an image batch with pixel-wise PSF patch by patch. Overlapping windows are used to avoid boundary artifacts.

    Args:
        img (Tensor): The image to be blurred (1, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, 3, ks, ks). ks can be odd or even.
        patch_num (list): Number of patches in each dimension. Defaults to (4, 4).
        expand (bool): Whether to expand image for the final output. Default is False.

    Returns:
        img_render (Tensor): Rendered image with same shape (1, C, H, W) as input. if expand is True, the output will be (1, C, H+pad*2, W+pad*2)
    """
    raise Exception("This function has not been tested.")
    B, Cimg, Himg, Wimg = img.shape
    Hpsf, Wpsf, Cpsf, _, ks = psf.shape
    assert B == 1, "Only support batch size 1"
    assert Cimg == Cpsf and Himg == Hpsf and Wimg == Wpsf, (
        "Input and PSF shape mismatch"
    )

    # Calculate base patch size and image padding
    patch_h, patch_w = patch_num
    base_patch_h = Himg // patch_h
    base_patch_w = Wimg // patch_w
    pad = int((ks - 1) / 2)

    # Initialize output and weight accumulation tensors
    img_render = torch.zeros_like(img)  # [1, C, Himg, Wimg]
    img_render = F.pad(
        img_render, (pad, pad, pad, pad), mode="reflect"
    )  # [1, C, Himg+pad*2, Wimg+pad*2]

    # Process each patch with overlap
    for pi in range(patch_h):
        for pj in range(patch_w):
            # Calculate patch boundaries with overlap
            low_i = pi * base_patch_h
            up_i = (pi + 1) * base_patch_h
            low_j = pj * base_patch_w
            up_j = (pj + 1) * base_patch_w

            # take care of the residual on last patch
            # for example, if Himg=100, patch_h=3, then the last patch will be [66:100] instead of [66:99]
            if pi == patch_h - 1:
                up_i = Himg
            if pj == patch_w - 1:
                up_j = Wimg

            # Extract patches
            img_patch = img[:, :, low_i:up_i, low_j:up_j]
            psf_patch = psf[low_i:up_i, low_j:up_j, :, :, :]

            # Process patch, expand boundary to [B, C, Himg+pad*2, Wimg+pad*2]
            rendered_patch = conv_psf_pixel(img_patch, psf_patch, expand=True)

            # Accumulate weighted result
            img_render[:, :, low_i : up_i + pad * 2, low_j : up_j + pad * 2] += (
                rendered_patch
            )

    if not expand:
        # Remove padding
        img_render = img_render[:, :, pad:-pad, pad:-pad]

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
        grid_h, grid_w, C, ks_h, ks_w = psf_map.shape
        assert grid_h == grid_w, f"PSF map grid must be square, got {grid_h}x{grid_w}"
        assert ks_h == ks_w, f"PSF kernel must be square, got {ks_h}x{ks_w}"
        grid_old = grid_h
        ks = ks_h
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


def rotate_psf(psf, theta):
    """Rotate PSF by theta counter-clockwise. Rotation center is the center of the PSF.

    Args:
        psf: (N, 3, ks, ks).
        theta: (N,). rotation angle in radians (counter-clockwise).

    Returns:
        rotated_psf: (N, 3, ks, ks).
    """
    assert len(psf.shape) == 4, "PSF should be [N, 3, ks, ks]"

    N, _, ks, _ = psf.shape
    assert ks == psf.shape[3], "PSF kernel should be square"

    # To rotate the image counter-clockwise, the sampling grid must be rotated clockwise.
    # The matrix for a clockwise rotation by theta is:
    # [ cos(theta)  sin(theta) ]
    # [ -sin(theta) cos(theta) ]
    rotation_matrices = torch.zeros(N, 2, 3, device=psf.device, dtype=psf.dtype)
    rotation_matrices[:, 0, 0] = torch.cos(theta)
    rotation_matrices[:, 0, 1] = torch.sin(theta)
    rotation_matrices[:, 1, 0] = -torch.sin(theta)
    rotation_matrices[:, 1, 1] = torch.cos(theta)

    # Rotate PSFs
    grid = F.affine_grid(rotation_matrices, psf.shape, align_corners=True)
    rotated_psf = F.grid_sample(psf, grid, align_corners=True)

    return rotated_psf


# ================================================
# Inverse PSF calculation from images
# ================================================
def solve_psf(img_org, img_render, ks=PSF_KS, eps=1e-6):
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


def solve_psf_map(img_org, img_render, ks=PSF_KS, grid=10):
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

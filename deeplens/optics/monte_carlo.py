# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Forward and backward Monte-Carlo integral functions."""

import torch
import torch.nn.functional as F

from deeplens.optics.basics import EPSILON


def forward_integral(ray, ps, ks, pointc=None, coherent=False):
    """Forward Monte-Carlo integral. Usecase example: PSF and Wavefront computation. This function donot do normalization.

    Args:
        ray: ray object. Shape of [N, spp, 3].
        ps: pixel size in [mm]
        ks: kernel size
        pointc: reference center point, shape [N, 2]
        coherent: whether or not coherent ray tracing

    Returns:
        field: intensity or complex amplitude, shape [N, ks, ks]
    """
    assert len(ray.o.shape) == 3, "Only support [N, spp, 3] shaped rays for now."
    points = -ray.o[..., :2]  # shape [N, spp, 2]. flip points.
    valid = ray.valid  # shape [N, spp]

    # Points shift relative to center
    if pointc is None:
        # Use ray spot center as PSF/Wavefront center if not specified
        pointc = (points * valid.unsqueeze(-1)).sum(-2) / valid.unsqueeze(-1).sum(-2).add(
            EPSILON
        )
    points_shift = points - pointc.unsqueeze(-2).repeat(1, points.shape[-2], 1)

    # Remove invalid points
    field_range = [
        -(ks / 2 - 0.5) * ps,
        (ks / 2 - 0.5) * ps,
    ]
    valid = (
        valid
        * (points_shift[..., 0].abs() < (field_range[1] - 0.001 * ps))
        * (points_shift[..., 1].abs() < (field_range[1] - 0.001 * ps))
    )  # shape [N, spp]
    points_shift = points_shift * valid.unsqueeze(-1)

    # Monte Carlo integral
    if not coherent:
        # Incoherent ray tracing, integral over intensity
        field = []
        for i in range(points.shape[0]):
            # Iterate over N points
            points_shift0 = points_shift[i]  # [spp, 2]
            valid0 = valid[i]  # [spp]
            amp = ray.d[i, :, 2] ** 2  # [spp]

            field0 = assign_points_to_pixels(
                points=points_shift0,
                mask=valid0,
                ks=ks,
                x_range=field_range,
                y_range=field_range,
                amp=amp,
            )
            field.append(field0)

        field = torch.stack(field, dim=0)  # shape [N, ks, ks]

    else:
        # Coherent ray tracing, integral over complex amplitude
        field = []
        for i in range(points.shape[0]):
            # Iterate over N points
            points_shift0 = points_shift[i]  # [spp, 2]
            valid0 = valid[i]  # [spp]
            amp = ray.d[i, :, 2]  # [spp]
            opl = ray.opl[i].squeeze(-1)  # [spp]
            wvln_mm = ray.wvln[i].squeeze(-1) * 1e-3  # [spp]
            phase = torch.fmod((opl - opl.min()) / wvln_mm, 1) * (2 * torch.pi)  # [spp]

            field_u = assign_points_to_pixels(
                points=points_shift0,
                mask=valid0,
                ks=ks,
                x_range=field_range,
                y_range=field_range,
                coherent=True,
                amp=amp,
                phase=phase,
            )
            field.append(field_u)

        field = torch.stack(field, dim=0)  # shape [N, ks, ks]

    return field


def assign_points_to_pixels(
    points,
    mask,
    ks,
    x_range,
    y_range,
    interpolate=True,
    coherent=False,
    amp=None,
    phase=None,
):
    """Assign points to pixels, supports both incoherent and coherent ray tracing. Use advanced indexing to increment the count for each corresponding pixel.

    This function can only compute single point source, constrained by advanced indexing operation.

    Args:
        points: shape [spp, 2]
        mask: shape [spp]
        ks: kernel size
        x_range: x range
        y_range: y range
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp], values we want to assign to each pixel
        amp: shape [spp], values we want to assign to each pixel (for incoherent ray tracing, typically 1)

    Returns:
        field: intensity or complex amplitude, shape [ks, ks]
    """
    # Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Normalize points to the range [0, 1]
    points_normalized = torch.zeros_like(points)
    points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
    points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

    # Check if points are within valid range
    valid_points = (points_normalized >= 0) & (points_normalized <= 1)
    valid_points = valid_points.all(dim=1)
    mask = mask * valid_points

    # If amp is not provided, set it to 1 (as default for incoherent ray tracing)
    if amp is None:
        amp = torch.ones_like(mask)

    if interpolate:
        # Compute weight (range [0, 1])
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[:, 0] - pixel_indices_float[:, 0].floor()
        w_r = pixel_indices_float[:, 1] - pixel_indices_float[:, 1].floor()

        # Compute pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = (
            torch.stack(
                (pixel_indices_float[:, 0], pixel_indices_float[:, 1] + 1), dim=-1
            )
            .floor()
            .long()
        )
        pixel_indices_bl = (
            torch.stack(
                (pixel_indices_float[:, 0] + 1, pixel_indices_float[:, 1]), dim=-1
            )
            .floor()
            .long()
        )
        pixel_indices_br = pixel_indices_tl + 1

        # Clamp indices to valid range
        pixel_indices_tl = torch.clamp(pixel_indices_tl, 0, ks - 1)
        pixel_indices_tr = torch.clamp(pixel_indices_tr, 0, ks - 1)
        pixel_indices_bl = torch.clamp(pixel_indices_bl, 0, ks - 1)
        pixel_indices_br = torch.clamp(pixel_indices_br, 0, ks - 1)

        # Use advanced indexing to increment the count for each corresponding pixel
        if coherent:
            if phase is None:
                raise ValueError("Phase must be provided for coherent mode")

            grid = torch.zeros(ks, ks, dtype=torch.complex128).to(device)
            grid.index_put_(
                tuple(pixel_indices_tl.t()),
                (1 - w_b) * (1 - w_r) * mask * amp * torch.exp(1j * phase),
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_tr.t()),
                (1 - w_b) * w_r * mask * amp * torch.exp(1j * phase),
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_bl.t()),
                w_b * (1 - w_r) * mask * amp * torch.exp(1j * phase),
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_br.t()),
                w_b * w_r * mask * amp * torch.exp(1j * phase),
                accumulate=True,
            )

        else:
            grid = torch.zeros(ks, ks).to(device)
            grid.index_put_(
                tuple(pixel_indices_tl.t()),
                (1 - w_b) * (1 - w_r) * mask * amp,
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_tr.t()),
                (1 - w_b) * w_r * mask * amp,
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_bl.t()),
                w_b * (1 - w_r) * mask * amp,
                accumulate=True,
            )
            grid.index_put_(
                tuple(pixel_indices_br.t()), w_b * w_r * mask * amp, accumulate=True
            )

    else:
        pixel_indices_float = points_normalized * (ks - 1)
        pixel_indices_tl = pixel_indices_float.floor().long()

        # Clamp indices to valid range
        pixel_indices_tl = torch.clamp(pixel_indices_tl, 0, ks - 1)

        if coherent:
            if phase is None:
                raise ValueError("Phase must be provided for coherent mode")

            grid = torch.zeros(ks, ks, dtype=torch.complex128).to(device)
            grid.index_put_(
                tuple(pixel_indices_tl.t()),
                mask * amp * torch.exp(1j * phase),
                accumulate=True,
            )
        else:
            grid = torch.zeros(ks, ks).to(device)
            grid.index_put_(tuple(pixel_indices_tl.t()), mask * amp, accumulate=True)

    return grid


def backward_integral(
    ray, img, ps, H, W, interpolate=True, pad=True, energy_correction=1
):
    """Backward integral, for ray tracing based rendering.

    NOTE: this function is currently not used and needs to be checked.

    Ignore:
        1. sub-pixel phase shiftment
        2. ray ampuity energy decay

        If we want to use this correction terms, use energy_corrention variable.

    Args:
        ray: Ray object. Shape of ray.o is [spp, 1, 3].
        img: [B, C, H, W]
        ps: pixel size
        H: image height
        W: image width
        interpolate: whether to interpolate
        pad: whether to pad the image
        energy_correction: whether to keep incident and output image total energy unchanged

    Returns:
        output: shape [B, C, H, W]
    """
    assert len(img.shape) == 4
    h, w, spp, _ = ray.o.shape
    p = ray.o[..., :2]  # shape [h, w, spp, 2]
    p = p.permute(2, 0, 1, 3)  # shape [spp, h, w, 2]

    if pad:
        img = F.pad(img, (1, 1, 1, 1), "replicate")

        # Convert ray positions to uv coordinates
        u = torch.clamp(W / 2 + p[..., 0] / ps, min=-0.99, max=W - 0.01)
        v = torch.clamp(H / 2 + p[..., 1] / ps, min=0.01, max=H + 0.99)

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1
    else:
        # Convert ray positions to uv coordinates
        u = torch.clamp(W / 2 + p[..., 0] / ps, min=0.01, max=W - 1.01)
        v = torch.clamp(H / 2 + p[..., 1] / ps, min=1.01, max=H - 0.01)

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long()
        idx_j = u.floor().long()

    # gradients are stored in weight parameters
    w_i = v - v.floor().long()
    w_j = u.ceil().long() - u

    if ray.coherent:
        raise Exception("Backward coherent integral needs to be checked.")

    else:
        if interpolate:  # Bilinear interpolation
            # img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], out_img shape [N, C, spp, H, W]
            out_img = img[..., idx_i, idx_j] * w_i * w_j
            out_img += img[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
            out_img += img[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
            out_img += img[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)

        else:
            out_img = img[..., idx_i, idx_j]

        # Monte-Carlo integration
        output = (torch.sum(out_img * ray.valid * energy_correction, -3) + 1e-9) / (
            torch.sum(ray.valid, -3) + 1e-6
        )
        return output

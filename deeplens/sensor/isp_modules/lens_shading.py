"""Lens shading correction (LSC).

Reference:
    [1] https://github.com/QiuJueqin/fast-openISP/blob/master/modules/lsc.py
    [2] Lens shading causes vignetting (darkening at image corners) due to
        optical properties of the lens. LSC compensates by applying a gain map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LensShadingCorrection(nn.Module):
    """Lens shading correction (LSC).

    Corrects vignetting (darkening at edges/corners) caused by lens optical
    properties by applying a spatially-varying gain map.
    """

    def __init__(self, shading_map=None, strength=1.0, falloff_model="radial"):
        """Initialize lens shading correction module.

        Args:
            shading_map: Pre-computed shading gain map of shape [H, W] or [1, 1, H, W].
                         If None, a radial falloff model is used. Default is None.
            strength: Strength of the correction (0-1). 0 = no correction, 1 = full. Default is 1.0.
            falloff_model: Model for computing gain map. Options: "radial", "polynomial".
                           Only used if shading_map is None. Default is "radial".
        """
        super().__init__()
        self.strength = strength
        self.falloff_model = falloff_model

        if shading_map is not None:
            if isinstance(shading_map, torch.Tensor):
                if shading_map.dim() == 2:
                    shading_map = shading_map.unsqueeze(0).unsqueeze(0)
                self.register_buffer("shading_map", shading_map)
            else:
                raise ValueError("shading_map must be a torch.Tensor")
        else:
            self.shading_map = None

        # Polynomial coefficients for vignetting model (typical values)
        # V(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6
        self.register_buffer("poly_coeffs", torch.tensor([0.3, 0.15, 0.05]))

    def _compute_radial_gain(self, H, W, device, dtype):
        """Compute radial gain map based on distance from center.

        Args:
            H, W: Height and width of the image.
            device: Target device.
            dtype: Target dtype.

        Returns:
            gain_map: Gain map of shape [1, 1, H, W].
        """
        # Create coordinate grid normalized to [-1, 1]
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Compute radial distance from center (normalized to [0, 1] at corners)
        r2 = (xx**2 + yy**2) / 2  # Normalize so corner is ~1

        if self.falloff_model == "radial":
            # Simple cos^4 falloff model (common approximation)
            # The gain is inverse of the falloff
            cos_theta = 1.0 / torch.sqrt(1 + r2)
            falloff = cos_theta**4
            gain = 1.0 / (falloff + 1e-6)

        elif self.falloff_model == "polynomial":
            # Polynomial vignetting model: V(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6
            r4 = r2**2
            r6 = r2**3
            k1, k2, k3 = self.poly_coeffs
            falloff = 1.0 - k1 * r2 - k2 * r4 - k3 * r6
            falloff = torch.clamp(falloff, min=0.1)  # Avoid division by zero
            gain = 1.0 / falloff

        else:
            raise ValueError(f"Unknown falloff model: {self.falloff_model}")

        # Normalize gain so center has gain of 1
        gain = gain / gain[H // 2, W // 2]

        return gain.view(1, 1, H, W)

    def forward(self, x):
        """Apply lens shading correction to remove vignetting.

        Args:
            x: Input tensor of shape [B, C, H, W], data range [0, 1].

        Returns:
            x_corrected: Corrected tensor of shape [B, C, H, W].
        """
        if self.strength == 0:
            return x

        B, C, H, W = x.shape

        # Get or compute the gain map
        if self.shading_map is not None:
            # Resize shading map to match input if needed
            if self.shading_map.shape[-2:] != (H, W):
                gain_map = F.interpolate(
                    self.shading_map, size=(H, W), mode="bilinear", align_corners=True
                )
            else:
                gain_map = self.shading_map
        else:
            # Compute gain map on-the-fly
            gain_map = self._compute_radial_gain(H, W, x.device, x.dtype)

        # Apply strength-weighted correction
        # gain = 1 + strength * (computed_gain - 1)
        effective_gain = 1 + self.strength * (gain_map - 1)

        # Apply correction
        x_corrected = x * effective_gain

        # Clamp to valid range
        x_corrected = torch.clamp(x_corrected, 0.0, 1.0)

        return x_corrected

    def reverse(self, x):
        """Reverse lens shading correction (add vignetting back).

        Args:
            x: Input tensor of shape [B, C, H, W], data range [0, 1].

        Returns:
            x_vignetted: Tensor with vignetting applied, shape [B, C, H, W].
        """
        if self.strength == 0:
            return x

        B, C, H, W = x.shape

        # Get or compute the gain map
        if self.shading_map is not None:
            if self.shading_map.shape[-2:] != (H, W):
                gain_map = F.interpolate(
                    self.shading_map, size=(H, W), mode="bilinear", align_corners=True
                )
            else:
                gain_map = self.shading_map
        else:
            gain_map = self._compute_radial_gain(H, W, x.device, x.dtype)

        # Compute inverse gain
        effective_gain = 1 + self.strength * (gain_map - 1)
        inverse_gain = 1.0 / effective_gain

        # Apply inverse correction (add vignetting)
        x_vignetted = x * inverse_gain

        return x_vignetted

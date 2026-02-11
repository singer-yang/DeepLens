"""Global tone mapping operators."""

import torch
import torch.nn as nn


class ToneMapping(nn.Module):
    """Global tone mapping operator.

    Maps HDR linear radiance values to displayable [0, 1] range using a
    global (per-pixel, spatially invariant) curve.

    Supported methods:
        - "reinhard":  L / (1 + L), from [Reinhard et al. 2002].
        - "aces":      ACES filmic curve approximation, from [Narkowicz 2015].
        - "hable":     Uncharted 2 filmic curve, from [Hable 2010].

    Reference:
        [1] Reinhard et al., "Photographic Tone Reproduction for Digital Images", SIGGRAPH 2002.
        [2] Narkowicz, "ACES Filmic Tone Mapping Curve", 2015.
        [3] Hable, "Filmic Tonemapping Operators", GDC 2010.
    """

    def __init__(self, method="reinhard", exposure=1.0):
        """Initialize tone mapping module.

        Args:
            method: Tone mapping method, one of "reinhard", "aces", "hable".
            exposure: Exposure multiplier applied before tone mapping.
        """
        super().__init__()
        if method not in ("reinhard", "aces", "hable"):
            raise ValueError(f"Unknown tone mapping method: {method}")
        self.method = method
        self.register_buffer("exposure", torch.tensor(exposure))

    def forward(self, img):
        """Apply global tone mapping.

        Args:
            img: HDR linear image, (B, C, H, W), range [0, +inf).

        Returns:
            img_tm: Tone-mapped image, (B, C, H, W), range [0, 1].
        """
        img = torch.clamp(img, min=0.0) * self.exposure

        if self.method == "reinhard":
            img_tm = img / (1.0 + img)
        elif self.method == "aces":
            img_tm = self._aces(img)
        elif self.method == "hable":
            img_tm = self._hable(img)

        return torch.clamp(img_tm, 0.0, 1.0)

    def reverse(self, img):
        """Inverse tone mapping (recover linear HDR from tone-mapped image).

        Only analytically invertible for "reinhard". For "aces" and "hable",
        uses an iterative Newton's method approximation.

        Args:
            img: Tone-mapped image, (B, C, H, W), range [0, 1].

        Returns:
            img_hdr: Recovered linear image, (B, C, H, W), range [0, +inf).
        """
        img = torch.clamp(img, 0.0, 1.0 - 1e-6)

        if self.method == "reinhard":
            img_hdr = img / (1.0 - img)
        elif self.method == "aces":
            img_hdr = self._aces_reverse(img)
        elif self.method == "hable":
            img_hdr = self._hable_reverse(img)

        return torch.clamp(img_hdr, min=0.0) / self.exposure

    # ---------------------------
    # ACES filmic curve
    # ---------------------------
    @staticmethod
    def _aces(x):
        """ACES filmic tone mapping curve.

        Attempt to approximate the ACES (Academy Color Encoding System) filmic curve.

        f(x) = (ax^2 + bx) / (cx^2 + dx + e)
             = x(ax + b) / (x(cx + d) + e)

        Reference:
            [1] Narkowicz, "ACES Filmic Tone Mapping Curve", 2015.
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return (x * (a * x + b)) / (x * (c * x + d) + e)

    @staticmethod
    def _aces_reverse(y):
        """Inverse ACES curve via quadratic formula.

        From  y = x(ax+b) / (x(cx+d)+e),  rearrange to
        (a - cy) x^2 + (b - dy) x - ey = 0
        and solve for x using the quadratic formula.
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        A = a - c * y
        B = b - d * y
        C = -e * y

        discriminant = torch.clamp(B**2 - 4 * A * C, min=0.0)
        # Take the positive root
        x = (-B + torch.sqrt(discriminant)) / (2 * A + 1e-8)
        return torch.clamp(x, min=0.0)

    # ---------------------------
    # Hable / Uncharted 2 curve
    # ---------------------------
    @staticmethod
    def _hable_partial(x):
        """Uncharted 2 tone mapping partial function.

        f(x) = ((x(Ax+CB)+DE) / (x(Ax+B)+DF)) - E/F

        Reference:
            [1] Hable, "Filmic Tonemapping Operators", GDC 2010.
        """
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    @classmethod
    def _hable(cls, x):
        """Hable / Uncharted 2 tone mapping with white point normalization."""
        W = 11.2  # linear white point
        return cls._hable_partial(x) / cls._hable_partial(torch.tensor(W, device=x.device))

    @classmethod
    def _hable_reverse(cls, y, num_iters=5):
        """Inverse Hable curve via Newton's method.

        Args:
            y: Tone-mapped values in [0, 1).
            num_iters: Number of Newton iterations.

        Returns:
            x: Approximate linear values.
        """
        # Initial guess from Reinhard inverse (reasonable starting point)
        x = y / (1.0 - y + 1e-6)
        for _ in range(num_iters):
            fx = cls._hable(x) - y
            # Numerical derivative
            eps = 1e-4
            dfx = (cls._hable(x + eps) - cls._hable(x - eps)) / (2 * eps)
            x = x - fx / (dfx + 1e-8)
            x = torch.clamp(x, min=0.0)
        return x

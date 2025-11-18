"""Pixel-based phase surface for diffractive optics."""

import torch
import torch.nn.functional as F

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class PixelPhase(Phase):
    """Phase profile using pixel-based parameterization.

    This class implements a diffractive surface where the phase profile is
    represented by a 2D tensor of phase values on a pixel grid. Phase values
    at arbitrary coordinates are obtained via bilinear interpolation.

    The pixel grid spans from -r to r in both x and y directions, discretized
    into a grid of shape (height, width).
    """

    def __init__(
        self,
        r,
        d,
        pixel_height=64,
        pixel_width=64,
        phase_map=None,
        norm_radii=None,
        mat2="air",
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=True,
        device="cpu",
    ):
        """Initialize pixel-based phase surface.

        Args:
            r: Radius of the surface
            d: Distance to next surface
            pixel_height: Number of pixels in y-direction (default: 64)
            pixel_width: Number of pixels in x-direction (default: 64)
            phase_map: Optional 2D tensor of shape (pixel_height, pixel_width)
                      with initial phase values in radians. If None, initialized
                      with random small values.
            norm_radii: Normalization radius (default: r)
            mat2: Material on the right side (default: "air")
            pos_xy: Position in xy plane
            vec_local: Local coordinate system vector
            is_square: Whether the aperture is square
            device: Computation device
        """
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # Pixel grid dimensions
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width

        # Initialize phase map
        if phase_map is None:
            # Initialize with random small values
            self.phase_map = torch.randn(pixel_height, pixel_width) * 1e-3
        else:
            self.phase_map = torch.tensor(phase_map, dtype=torch.float32)
            assert self.phase_map.shape == (pixel_height, pixel_width), (
                f"phase_map must have shape ({pixel_height}, {pixel_width})"
            )

        self.to(device)
        self.init_param_model()

    def init_param_model(self):
        """Initialize pixel parameters."""
        self.param_model = "pixel"
        self.to(self.device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize pixel phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        pixel_height = surf_dict.get("pixel_height", 64)
        pixel_width = surf_dict.get("pixel_width", 64)

        obj = cls(
            surf_dict["r"],
            surf_dict["(d)"],
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # Load phase map
        phase_map = surf_dict.get("phase_map", None)
        if phase_map is not None:
            obj.phase_map = torch.tensor(phase_map, device=obj.device)

        return obj

    def phi(self, x, y):
        """Reference phase map at design wavelength using pixel interpolation.

        Uses bilinear interpolation to sample the phase map at the given coordinates.

        Args:
            x, y: Coordinate tensors

        Returns:
            Phase values in radians at the specified coordinates
        """
        # Normalize coordinates to [-1, 1] range
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # Flatten input coordinates for batch processing
        x_flat = x_norm.flatten()
        y_flat = y_norm.flatten()
        original_shape = x_norm.shape

        # Create sampling grid for bilinear interpolation
        # grid_sample expects grid of shape (N, H_out, W_out, 2) where H_out=W_out=1 for point sampling
        # Coordinates should be in [-1, 1] range where (-1, -1) is top-left and (1, 1) is bottom-right
        # Our coordinate system has (0,0) at center, so we map accordingly

        batch_size = x_flat.shape[0]
        grid = torch.zeros(
            batch_size, 1, 1, 2, dtype=x_flat.dtype, device=x_flat.device
        )

        # Set x and y coordinates (flip y for grid_sample convention)
        grid[..., 0, 0, 0] = x_flat  # x coordinate
        grid[..., 0, 0, 1] = -y_flat  # y coordinate (flipped)

        # Prepare phase map for grid_sample: add channel dimension
        # Shape: (1, 1, pixel_height, pixel_width)
        phase_map_expanded = self.phase_map.unsqueeze(0).unsqueeze(0)

        # Expand phase map to match batch size
        phase_map_batch = phase_map_expanded.expand(batch_size, -1, -1, -1)

        # Perform bilinear interpolation
        sampled_phase = F.grid_sample(
            phase_map_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Remove extra dimensions and reshape
        phi = sampled_phase.squeeze(-1).squeeze(-1).squeeze(1)  # Shape: (batch_size,)
        phi = phi.reshape(original_shape)

        # Apply circular aperture mask (set phase to 0 outside unit circle)
        r_squared = x_norm**2 + y_norm**2
        mask = r_squared > 1
        phi = torch.where(mask, torch.zeros_like(phi), phi)

        # Ensure phase is in [0, 2π) range
        phi = torch.remainder(phi, 2 * torch.pi)

        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) using pixel interpolation.

        Computes derivatives of the bilinearly interpolated phase field.

        Args:
            x, y: Coordinate tensors

        Returns:
            dphidx, dphidy: Phase derivatives in x and y directions
        """
        # Normalize coordinates
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # For bilinear interpolation, the derivative can be computed by
        # taking the difference of interpolated values at slightly offset positions

        eps = 1e-6  # Small offset for numerical differentiation

        # Compute phi at (x+eps, y) and (x-eps, y) for dphi/dx
        phi_x_plus = self.phi(x + eps, y)
        phi_x_minus = self.phi(x - eps, y)
        dphidx = (phi_x_plus - phi_x_minus) / (2 * eps)

        # Compute phi at (x, y+eps) and (x, y-eps) for dphi/dy
        phi_y_plus = self.phi(x, y + eps)
        phi_y_minus = self.phi(x, y - eps)
        dphidy = (phi_y_plus - phi_y_minus) / (2 * eps)

        # Apply circular mask
        r_squared = x_norm**2 + y_norm**2
        mask = r_squared > 1
        dphidx = torch.where(mask, torch.zeros_like(dphidx), dphidx)
        dphidy = torch.where(mask, torch.zeros_like(dphidy), dphidy)

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Generate optimizer parameters for pixel phase map."""
        params = []

        # Enable gradients for the phase map
        self.phase_map.requires_grad = True
        params.append({"params": [self.phase_map], "lr": lrs[0]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./pixel_doe.pth"):
        """Save pixel DOE parameters."""
        torch.save(
            {
                "param_model": "pixel",
                "phase_map": self.phase_map.clone().detach().cpu(),
                "pixel_height": self.pixel_height,
                "pixel_width": self.pixel_width,
            },
            save_path,
        )

    def load_ckpt(self, load_path="./pixel_doe.pth"):
        """Load pixel DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.phase_map = ckpt["phase_map"].to(self.device)
        self.pixel_height = ckpt["pixel_height"]
        self.pixel_width = ckpt["pixel_width"]

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": "Phase",
            "r": self.r,
            "is_square": self.is_square,
            "param_model": "pixel",
            "phase_map": self.phase_map.clone().detach().cpu().tolist(),
            "pixel_height": self.pixel_height,
            "pixel_width": self.pixel_width,
            "norm_radii": round(self.norm_radii, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

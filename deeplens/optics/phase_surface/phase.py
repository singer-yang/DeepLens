"""Phase class: a plane surface with phase pattern on it."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deeplens.basics import EPSILON, DeepObj
from deeplens.optics.materials import Material


class Phase(DeepObj):
    """Base phase profile for diffractive surfaces (metasurface or DOE).

    This is the base class that provides common functionality for all phase parameterizations.
    Specific parameterizations should inherit from this class.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=True,
        device="cpu",
    ):
        super().__init__()

        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]

        # Global direction vector, always pointing to the positive z-axis
        self.vec_global = torch.tensor([0.0, 0.0, 1.0])

        # Surface position in global coordinate system
        self.d = torch.tensor(d)
        self.pos_x = torch.tensor(pos_xy[0])
        self.pos_y = torch.tensor(pos_xy[1])

        # Surface direction vector in global coordinate system
        self.vec_local = F.normalize(torch.tensor(vec_local), p=2, dim=-1)

        # Material after the surface
        self.mat2 = Material(mat2)

        # DOE geometry
        self.r = float(r)
        self.is_square = is_square
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

        # Use ray tracing to simulate diffraction, the same as Zemax
        self.diffraction = True
        self.diffraction_order = 1
        self.norm_radii = self.r if norm_radii is None else norm_radii

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

    # ==============================
    # Abstract methods to be implemented by subclasses
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength. Must be implemented by subclasses."""
        raise NotImplementedError("phi() must be implemented by subclasses")

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives. Must be implemented by subclasses."""
        raise NotImplementedError("dphi_dxy() must be implemented by subclasses")

    def init_param_model(self):
        """Initialize parameterization parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "init_param_model() must be implemented by subclasses"
        )

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "get_optimizer_params() must be implemented by subclasses"
        )

    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("save_ckpt() must be implemented by subclasses")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("load_ckpt() must be implemented by subclasses")

    def surf_dict(self):
        """Return surface parameters. Must be implemented by subclasses."""
        raise NotImplementedError("surf_dict() must be implemented by subclasses")

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def activate_diffraction(self, diffraction_order=1):
        """Activate diffraction of DOE in ray tracing."""
        self.diffraction = True
        self.diffraction_order = diffraction_order
        print("Diffraction of DOE in ray tracing is enabled.")

    def ray_reaction(self, ray, n1=None, n2=None):
        """Ray reaction on DOE surface."""
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.refract(ray, n1 / n2)
        if self.diffraction:
            ray = self.diffract(ray)
        ray = self.to_global_coord(ray)
        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-plane intersection in local coordinate system and update ray data."""
        # Solve intersection
        t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.is_valid > 0)
            )
        else:
            valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
                ray.is_valid > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.coherent:
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def diffract(self, ray):
        """Diffraction of DOE surface.
            1, The phase φ in radians adds to the optical path length of the ray
            2, The gradient of the phase profile (phase slope) change the direction of rays.

        Reference:
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 2011.
        """
        forward = (ray.d * ray.is_valid.unsqueeze(-1))[..., 2].sum() > 0
        valid = ray.is_valid > 0

        # Diffraction 1: DOE phase modulation
        if ray.coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1])
            new_opl = ray.opl + phi.unsqueeze(-1) * (ray.wvln * 1e-3) / (2 * torch.pi)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        # Diffraction 2: bend rays
        # Perpendicular incident rays are diffracted following (1) grating equation and (2) local grating approximation
        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

        wvln_mm = ray.wvln * 1e-3
        order = self.diffraction_order
        if forward:
            new_d_x = ray.d[..., 0] + wvln_mm / (2 * torch.pi) * dphidx * order
            new_d_y = ray.d[..., 1] + wvln_mm / (2 * torch.pi) * dphidy * order
        else:
            new_d_x = ray.d[..., 0] - wvln_mm / (2 * torch.pi) * dphidx * order
            new_d_y = ray.d[..., 1] - wvln_mm / (2 * torch.pi) * dphidy * order

        new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        return ray

    def refract(self, ray, eta):
        """Calculate refracted ray according to Snell's law in local coordinate system.

        Args:
            ray (Ray): incident ray.
            eta (float): ratio of indices of refraction, eta = n_i / n_t

        Returns:
            ray (Ray): refracted ray.
        """
        # Compute normal vectors
        normal_vec = self.normal_vec(ray)

        # Compute refraction according to Snell's law
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2)

        # Total internal reflection
        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        # Update ray direction
        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        # Update ray valid mask
        ray.is_valid = ray.is_valid * valid

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points.

        Normal vector points from the surface toward the side where the light is coming from.
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1
        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def to_local_coord(self, ray):
        """Transform ray to local coordinate system.

        Args:
            ray (Ray): input ray in global coordinate system.

        Returns:
            ray (Ray): transformed ray in local coordinate system.
        """
        # Shift ray origin to surface origin
        ray.o[..., 0] = ray.o[..., 0] - self.pos_x
        ray.o[..., 1] = ray.o[..., 1] - self.pos_y
        ray.o[..., 2] = ray.o[..., 2] - self.d

        # Rotate ray origin and direction
        if torch.abs(torch.dot(self.vec_local, self.vec_global) - 1.0) > EPSILON:
            R = self._get_rotation_matrix(self.vec_local, self.vec_global)
            ray.o = self._apply_rotation(ray.o, R)
            ray.d = self._apply_rotation(ray.d, R)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        return ray

    def to_global_coord(self, ray):
        """Transform ray to global coordinate system.

        Args:
            ray (Ray): input ray in local coordinate system.

        Returns:
            ray (Ray): transformed ray in global coordinate system.
        """
        # Rotate ray origin and direction
        if torch.abs(torch.dot(self.vec_local, self.vec_global) - 1.0) > EPSILON:
            R = self._get_rotation_matrix(self.vec_global, self.vec_local)
            ray.o = self._apply_rotation(ray.o, R)
            ray.d = self._apply_rotation(ray.d, R)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        # Shift ray origin back to global coordinates
        ray.o[..., 0] = ray.o[..., 0] + self.pos_x
        ray.o[..., 1] = ray.o[..., 1] + self.pos_y
        ray.o[..., 2] = ray.o[..., 2] + self.d

        return ray

    def _get_rotation_matrix(self, vec_from, vec_to):
        """Calculate rotation matrix to rotate vec_from to vec_to."""
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        K = torch.tensor(
            [
                [0, -v_cross_u[2], v_cross_u[1]],
                [v_cross_u[2], 0, -v_cross_u[0]],
                [-v_cross_u[1], v_cross_u[0], 0],
            ],
            device=self.device,
        )

        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """Apply rotation matrix to vectors."""
        original_shape = vectors.shape
        vectors_flat = vectors.view(-1, 3)
        rotated_flat = torch.mm(vectors_flat, R.t())
        return rotated_flat.view(original_shape)

    def surface_with_offset(self, *args, **kwargs):
        """Surface sag with offset, only used in layout drawing."""
        return self.d

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer(self, lrs):
        """Generate optimizer.

        Args:
            lrs (list or float): Learning rates for different parameters.
        """
        if isinstance(lrs, float):
            lrs = [lrs]
        assert self.diffraction, "Diffraction is not activated yet."
        params = self.get_optimizer_params(lrs)
        optimizer = torch.optim.Adam(params)
        return optimizer

    # =========================================
    # Visualization
    # =========================================
    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2*pi]."""
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, 2000),
            torch.linspace(self.h / 2, -self.h / 2, 2000),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * torch.pi)
        ax.set_title("Phase map 0.55um", fontsize=10)
        ax.grid(False)
        fig.colorbar(im)
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="black", linestyle="-"):
        """Draw DOE as a two-side surface."""
        max_offset = self.d.item() / 100
        d = self.d.item()

        # Draw DOE
        roc = self.r * 2
        x = np.linspace(-self.r, self.r, 128)
        y = np.zeros_like(x)
        r = np.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - np.sqrt(1 - r**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color="orange", linestyle=linestyle, linewidth=0.75)

    # =========================================
    # IO
    # =========================================

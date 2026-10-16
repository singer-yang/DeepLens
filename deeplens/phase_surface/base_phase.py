"""Phase class: a plane substrate with phase pattern on it."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..base import DeepObj
from ..config import EPSILON
from ..material import Material


class Phase(DeepObj):
    """Base phase profile for diffractive surfaces (metasurface or DOE).

    Represents a flat (zero-sag) substrate carrying a phase pattern $\\phi(x, y)$,
    using sequential geometry. It owns only `d_next`, the thickness to the next
    vertex; the containing `GeoLens` derives its absolute position. Provides the
    common ray-tracing machinery (intersection, refraction, generalized-Snell
    diffraction, local/global transforms); the phase profile $\\phi$ and its
    gradient are defined by subclasses.

    Attributes:
        vec_global (torch.Tensor): Global axis direction $[0, 0, 1]$, shape [3].
        d_next (torch.Tensor): Thickness to the next vertex in [mm], scalar.
        pos_x (torch.Tensor): Surface x-offset in [mm], scalar.
        pos_y (torch.Tensor): Surface y-offset in [mm], scalar.
        vec_local (torch.Tensor): Unit surface normal in global coordinates, shape [3].
        mat2 (Material): Material on the exit side of the surface.
        r (float): Surface radius / half-aperture in [mm].
        is_square (bool): If True the aperture is a square of side $r\\sqrt{2}$;
            otherwise a circle of radius $r$.
        w (float): Square aperture width $r\\sqrt{2}$ in [mm].
        h (float): Square aperture height $r\\sqrt{2}$ in [mm].
        diffraction_order (int): Diffraction order $m$ used in the generalized
            Snell's law. Defaults to 1.
        norm_radii (float): Radius in [mm] used to normalize coordinates for the
            phase polynomial. Defaults to `r`.
        device (str or torch.device): Device holding the tensor state.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d_next,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a flat phase substrate.

        Args:
            r (float): Surface radius / half-aperture in [mm].
            d_next (float): Axial thickness to the next vertex in [mm].
            norm_radii (float or None, optional): Radius in [mm] used to normalize
                coordinates for the phase polynomial. Defaults to None, which uses `r`.
            mat2 (str, optional): Material on the exit side of the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) offset of the surface center in [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Surface normal direction (not necessarily
                normalized) in global coordinates. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True the aperture is a square of side
                $r\\sqrt{2}$; otherwise a circle of radius $r$. Defaults to True.
            device (str, optional): Device for the tensor state. Defaults to "cpu".
        """
        super().__init__()

        self.d_next = (
            d_next.detach().clone()
            if torch.is_tensor(d_next)
            else torch.tensor(d_next, dtype=torch.get_default_dtype())
        )
        if not self.d_next.is_floating_point():
            self.d_next = self.d_next.to(torch.get_default_dtype())

        state_dtype = self.d_next.dtype
        state_device = self.d_next.device
        self.vec_global = torch.tensor(
            [0.0, 0.0, 1.0], dtype=state_dtype, device=state_device
        )
        self.pos_x = torch.as_tensor(pos_xy[0], dtype=state_dtype, device=state_device)
        self.pos_y = torch.as_tensor(pos_xy[1], dtype=state_dtype, device=state_device)
        self.vec_local = F.normalize(
            torch.as_tensor(vec_local, dtype=state_dtype, device=state_device),
            p=2,
            dim=-1,
        )

        self.mat2 = Material(mat2)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.w = self.r * float(np.sqrt(2))
            self.h = self.r * float(np.sqrt(2))

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)
        self._cache_rotation_matrices()

        # Phase surfaces retain rectangular extents even for circular apertures
        # because phase-map visualization and fabrication helpers use them.
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

        self.diffraction_order = 1
        self.norm_radii = self.r if norm_radii is None else norm_radii

    def _get_effective_d_next(self):
        """Return the thickness used by the sequential lens tracer."""
        return self.d_next

    def _cache_rotation_matrices(self):
        """Pre-compute local/global rotation matrices for a static orientation."""
        needs_rotation = (
            torch.abs(torch.dot(self.vec_local, self.vec_global) - 1.0) > EPSILON
        )
        if needs_rotation:
            self._R_to_local = self._get_rotation_matrix(
                self.vec_local, self.vec_global
            )
            self._R_to_global = self._get_rotation_matrix(
                self.vec_global, self.vec_local
            )
        else:
            self._R_to_local = None
            self._R_to_global = None

    # ==============================
    # Abstract methods to be implemented by subclasses
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength. Must be implemented by subclasses."""
        raise NotImplementedError("phi() must be implemented by subclasses")

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives. Must be implemented by subclasses."""
        raise NotImplementedError("dphi_dxy() must be implemented by subclasses")

    def paraxial_power(self, n1, n2, wvln):
        """Return the paraxial optical power of the phase [1/mm].

        Near the axis `diffract` tilts a ray at height $h$ by
        $m\\,\\lambda/(2\\pi)\\,\\partial\\phi/\\partial x$, which is a thin lens of
        power $-m\\,\\lambda/(2\\pi)\\,\\partial^2\\phi/\\partial x^2$. The curvature is
        a central difference of `dphi_dxy` at $h = 10^{-3}$ `norm_radii`, averaged
        over $x$ and $y$, so it is the power the tracer applies for every phase
        profile (a Fresnel lens gives $m\\,\\lambda/(\\lambda_0 f_0)$; a grating or
        cubic phase gives 0). The flat substrate adds no power.

        Args:
            n1 (torch.Tensor): Refractive index of the incident medium (unused).
            n2 (torch.Tensor): Refractive index of the transmission medium (unused).
            wvln (float): Wavelength [µm].

        Returns:
            power (torch.Tensor): Surface power [1/mm], scalar.
        """
        h = 1e-3 * self.norm_radii
        s = torch.tensor([h, -h], dtype=self.d_next.dtype, device=self.d_next.device)
        dphidx, _ = self.dphi_dxy(s, torch.zeros_like(s))
        _, dphidy = self.dphi_dxy(torch.zeros_like(s), s)
        curvature = (dphidx[0] - dphidx[1] + dphidy[0] - dphidy[1]) / (4 * h)
        return -self.diffraction_order * wvln * 1e-3 / (2 * torch.pi) * curvature

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def ray_reaction(self, ray, n1, n2):
        """Trace a ray through the phase surface.

        Transforms the ray to local coordinates, intersects it with the plane,
        applies refraction then diffraction, and transforms back to global
        coordinates.

        Args:
            ray (Ray): Incident ray in global coordinates.
            n1 (float): Refractive index of the medium before the surface.
            n2 (float): Refractive index of the medium after the surface.

        Returns:
            ray (Ray): Updated ray in global coordinates.
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray, n1)
        ray = self.refract(ray, n1 / n2)
        ray = self.diffract(ray, n2=n2)
        ray = self.to_global_coord(ray)
        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-plane intersection in local coordinates and update the ray.

        Advances each ray to the $z = 0$ plane, marks rays falling outside the
        aperture as invalid, and (for coherent rays) accumulates optical path
        length. Rays nearly parallel to the plane are guarded against division
        by a near-zero z-direction.

        Args:
            ray (Ray): Ray in local coordinates.
            n (float, optional): Refractive index of the medium the ray travels
                through, used for OPL accumulation. Defaults to 1.0.

        Returns:
            ray (Ray): Ray advanced to the surface plane with updated `o`,
                `is_valid`, and `opl`.
        """
        # Solve intersection. Guard against a near-zero z-direction (rays
        # parallel to the plane) before dividing, matching ray.py prop_to.
        dz = ray.d[..., 2]
        dz = torch.where(dz.abs() < EPSILON, torch.full_like(dz, EPSILON), dz)
        t = (0.0 - ray.o[..., 2]) / dz
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.is_valid > 0)
            )
        else:
            valid = (new_o[..., 0] ** 2 + new_o[..., 1] ** 2 < self.r**2) & (
                ray.is_valid > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def diffract(self, ray, n2=1.0):
        """Apply phase-surface diffraction to a ray.

        Two effects are applied:

        1. The phase $\\phi$ (in radians) adds to the optical path length as
           $m\\,\\phi \\cdot \\lambda / (2\\pi)$, where $\\lambda$ is converted
           from [µm] to [mm] internally.
        2. The phase gradient bends the ray by the vector grating equation on
           the plane (normal $z$),
           $n_2 \\mathbf{t}_2 = n_1 \\mathbf{t}_1 + m\\,\\lambda / (2\\pi)\\,\\nabla\\phi$,
           for the tangential direction components $\\mathbf{t} = (d_x, d_y)$.
           Refraction is already applied, so the tangential components gain
           $m\\,\\lambda / (2\\pi n_2)\\,\\nabla\\phi$ and $d_z$ is recomputed so
           $|\\mathbf{d}| = 1$. An order with $|\\mathbf{t}_2| \\geq 1$ is
           evanescent and the ray is marked invalid. The relation is the same in
           both trace directions, so a backward trace retraces the forward path.

        Args:
            ray (Ray): Ray with position, direction, and wavelength in [µm].
            n2 (float, optional): Refractive index of the medium after the surface;
                the deflection scales as $1/n_2$. Defaults to 1.0.

        Returns:
            ray (Ray): Ray with updated direction `d`, `is_valid` (evanescent
                orders cleared) and (for coherent rays) `opl`.

        Note:
            Material dispersion is not modelled here. The phase profile $\\phi(x, y)$
            is treated as wavelength-independent; only the $\\lambda$ scaling in the
            generalized Snell's law and the OPL accumulation vary with wavelength.
            For a physical DOE whose phase profile itself changes with wavelength
            (via $(n(\\lambda) - 1)\\,h$), use `DiffractiveSurface` instead.

        Reference:
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 2011.
        """
        valid = ray.is_valid > 0
        order = self.diffraction_order

        # Step 1: DOE phase modulation
        if ray.is_coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1])
            new_opl = ray.opl + order * phi.unsqueeze(-1) * (ray.wvln * 1e-3) / (
                2 * torch.pi
            )
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        # Step 2: vector grating equation (see docstring).
        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])
        scale = order * ray.wvln * 1e-3 / (2 * torch.pi * n2)
        tx = ray.d[..., 0] + scale * dphidx
        ty = ray.d[..., 1] + scale * dphidy
        # Decide validity on detached values. Invalid rays get a zero tangent so
        # the masked sqrt never sees a negative radicand (NaN gradient); the
        # clamp keeps its gradient finite as |t| -> 1.
        with torch.no_grad():
            t2 = tx * tx + ty * ty
        valid = valid & torch.isfinite(t2) & (t2 < 1.0)
        tx = torch.where(valid, tx, torch.zeros_like(tx))
        ty = torch.where(valid, ty, torch.zeros_like(ty))
        dz = torch.sign(ray.d[..., 2]) * torch.sqrt(
            (1.0 - tx * tx - ty * ty).clamp_min(EPSILON)
        )
        new_d = torch.stack([tx, ty, dz], dim=-1)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)
        ray.is_valid = ray.is_valid * valid

        return ray

    def refract(self, ray, eta):
        """Refract a ray with vector Snell's law in local coordinates."""
        normal_vec = self.normal_vec(ray)
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2)

        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)
        ray.is_valid = ray.is_valid * valid
        return ray

    def to_local_coord(self, ray):
        """Transform a ray from the surface reference frame to local coordinates."""
        offset = torch.stack(
            [self.pos_x, self.pos_y, torch.zeros_like(self.pos_x)]
        ).expand_as(ray.o)
        ray.o = ray.o - offset

        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)
        return ray

    def to_global_coord(self, ray):
        """Transform a ray from local coordinates to the surface reference frame."""
        if self._R_to_global is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_global)
            ray.d = self._apply_rotation(ray.d, self._R_to_global)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        offset = torch.stack(
            [self.pos_x, self.pos_y, torch.zeros_like(self.pos_x)]
        ).expand_as(ray.o)
        ray.o = ray.o + offset
        return ray

    def _get_rotation_matrix(self, vec_from, vec_to):
        """Return the dtype/device-preserving rotation from one vector to another."""
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            return torch.eye(3, device=self.device, dtype=vec_from.dtype)

        if torch.abs(dot_product + 1.0) < EPSILON:
            if torch.abs(vec_from[0]) < 0.9:
                perpendicular = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=self.device,
                    dtype=vec_from.dtype,
                )
            else:
                perpendicular = torch.tensor(
                    [0.0, 1.0, 0.0],
                    device=self.device,
                    dtype=vec_from.dtype,
                )
            axis = F.normalize(torch.linalg.cross(vec_from, perpendicular), p=2, dim=-1)
            return 2.0 * torch.outer(axis, axis) - torch.eye(
                3, device=self.device, dtype=axis.dtype
            )

        cross_product = torch.linalg.cross(vec_from, vec_to)
        zero = torch.zeros((), device=self.device, dtype=cross_product.dtype)
        skew = torch.stack(
            [
                torch.stack([zero, -cross_product[2], cross_product[1]]),
                torch.stack([cross_product[2], zero, -cross_product[0]]),
                torch.stack([-cross_product[1], cross_product[0], zero]),
            ]
        )
        identity = torch.eye(3, device=self.device, dtype=skew.dtype)
        return identity + skew + torch.mm(skew, skew) / (1 + dot_product)

    @staticmethod
    def _apply_rotation(vectors, rotation):
        """Apply a rotation matrix to tensors whose final dimension is three."""
        original_shape = vectors.shape
        vectors_flat = vectors.reshape(-1, 3)
        rotated_flat = torch.mm(vectors_flat, rotation.t())
        return rotated_flat.reshape(original_shape)

    def normal_vec(self, ray):
        """Calculate the surface normal vector at intersection points.

        The normal points from the surface toward the side the light comes from
        (i.e. it is flipped to oppose the ray's z-direction).

        Args:
            ray (Ray): Ray providing the propagation direction.

        Returns:
            normal_vec (torch.Tensor): Unit normal vectors, same shape as `ray.d`.
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1
        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "get_optimizer_params() must be implemented by subclasses"
        )

    def get_optimizer(self, lrs):
        """Build an Adam optimizer over the surface's learnable parameters.

        Args:
            lrs (list or float): Learning rate(s) for the parameter groups. A
                single float is wrapped into a one-element list.

        Returns:
            optimizer (torch.optim.Adam): Adam optimizer over the parameters
                returned by `get_optimizer_params`.
        """
        if isinstance(lrs, float):
            lrs = [lrs]
        params = self.get_optimizer_params(lrs)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def update_r(self, r):
        """Update surface radius / half-aperture and the square aperture extents.

        A flat phase surface has no geometric height constraint, and because the
        polynomial is normalized by a fixed `norm_radii`, phase coefficients do
        not need rescaling.

        Args:
            r (float): New surface radius / half-aperture in [mm].
        """
        self.r = float(r)
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

    def phase2height_map(self, design_wvln, refractive_idx=1.5, res=512):
        """Convert the phase map to a physical height map for DOE fabrication.

        Derived from the phase-height relation of a transmissive DOE in air,
        $\\phi = (2\\pi/\\lambda)(n - 1)h$, giving $h = \\phi\\lambda / (2\\pi(n - 1))$.

        Args:
            design_wvln (float): Design wavelength in [µm].
            refractive_idx (float, optional): Refractive index of the DOE material
                at `design_wvln`. Defaults to 1.5.
            res (int, optional): Pixel resolution of the returned square height map.
                Defaults to 512.

        Returns:
            height_map (torch.Tensor): Height map of shape [res, res] in the same
                units as `design_wvln` ([µm]).
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, res),
            torch.linspace(self.h / 2, -self.h / 2, res),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        phi = self.phi(x, y)  # [0, 2π], shape [res, res]
        height_map = phi * design_wvln / (2 * torch.pi * (refractive_idx - 1))
        return height_map

    # =========================================
    # Visualization
    # =========================================
    def draw_r(self):
        """Effective drawing radius for 2D layout drawing."""
        return self.r

    def surface_with_offset(self, *args, d=0.0, **kwargs):
        """Return the caller-provided vertex position for layout drawing.

        The phase surface is flat and does not own an absolute position.

        Returns:
            d (torch.Tensor): Axial plane position in [mm], scalar.
        """
        if torch.is_tensor(d):
            return d
        return torch.tensor(float(d), device=self.device)

    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw the phase map (clipped to $[0, 2\\pi]$) and save it to a file.

        Args:
            save_name (str, optional): Output image path. Defaults to
                "./DOE_phase_map.png".
        """
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

    def draw_widget(self, ax, color="black", linestyle="-", d=0.0):
        """Draw the DOE as a sawtooth (blazed) profile on a 2D layout axis.

        Args:
            ax (matplotlib.axes.Axes): Axis to draw on.
            color (str, optional): Accepted for API consistency but ignored; the
                profile is always drawn in orange. Defaults to "black".
            linestyle (str, optional): Matplotlib line style for the profile.
                Defaults to "-".
        """
        max_offset = self.r / 100
        d = float(d)

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
    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("save_ckpt() must be implemented by subclasses")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("load_ckpt() must be implemented by subclasses")

    def surf_dict(self):
        """Return surface parameters. Must be implemented by subclasses."""
        raise NotImplementedError("surf_dict() must be implemented by subclasses")

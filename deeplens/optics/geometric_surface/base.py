"""Base class for geometric surfaces.

Surface can refract, and reflect rays. Some surfaces can also diffract rays according to a local grating approximation.
"""

import numpy as np
import torch
import torch.nn.functional as F

from deeplens.basics import DeepObj
from deeplens.optics.materials import Material

EPSILON = 1e-12


class Surface(DeepObj):
    def __init__(
        self,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        super(Surface, self).__init__()

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

        # Surface aperture radius (non-differentiable)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            # r is the incircle radius
            self.h = 2 * self.r
            self.w = 2 * self.r

        # Newton method parameters
        self.newton_maxiter = 10  # [int], maximum number of Newton iterations
        self.newton_convergence = (
            50.0 * 1e-6
        )  # [mm], Newton method convergence threshold
        self.newton_step_bound = self.r / 5  # [mm], maximum step size in each iteration

        self.tolerancing = False
        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize surface from a dict."""
        raise NotImplementedError(
            f"init_from_dict() is not implemented for {cls.__name__}."
        )

    # =====================================================================
    # Intersection, refraction, reflection between ray and surface
    # =====================================================================
    def ray_reaction(self, ray, n1, n2, refraction=True):
        """Compute output ray after intersection and refraction with a surface."""
        # Transform ray to local coordinate system
        ray = self.to_local_coord(ray)

        # Intersection
        ray = self.intersect(ray, n1)

        if refraction:
            # Refraction
            ray = self.refract(ray, n1 / n2)
        else:
            # Reflection
            ray = self.reflect(ray)

        # Transform ray to global coordinate system
        ray = self.to_global_coord(ray)

        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection in local coordinate system.

        Args:
            ray (Ray): input ray.
            n (float, optional): refractive index. Defaults to 1.0.
        """
        # Solve ray-surface intersection time by Newton's method
        t, valid = self.newtons_method(ray)

        # Update ray
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.coherent:
            if t.abs().max() > 100 and torch.get_default_dtype() == torch.float32:
                raise Exception(
                    "Using float32 may cause precision problem for OPL calculation."
                )
            new_opl = ray.opl + n * t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def newtons_method(self, ray):
        """Solve intersection by Newton's method in local coordinate system.

        Args:
            ray (Ray): input ray.

        Returns:
            t (tensor): intersection time.
            valid (tensor): valid mask.
        """
        newton_maxiter = self.newton_maxiter
        newton_convergence = self.newton_convergence
        newton_step_bound = self.newton_step_bound

        # Tolerance
        if self.tolerancing:
            d_surf = self.d_error
        else:
            d_surf = 0.0

        # Initial guess of t (can also use spherical surface for initial guess)
        t = - ray.o[..., 2] / ray.d[..., 2]

        # 1. Non-differentiable Newton's iterations to find the intersection points
        with torch.no_grad():
            it = 0
            ft = 1e6 * torch.ones_like(ray.o[..., 2])
            while it < newton_maxiter:
                # Converged
                if (torch.abs(ft) < newton_convergence).all():
                    break

                # One Newton step
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[..., 0], new_o[..., 1]
                valid = self.is_within_data_range(new_x, new_y) & (ray.is_valid > 0)

                ft = self.sag(new_x, new_y, valid) - new_o[..., 2]
                dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y, valid)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(
                    ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
                )

        # 2. One more (differentiable) Newton step to gain gradients
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[..., 0], new_o[..., 1]
        valid = self.is_valid(new_x, new_y) & (ray.is_valid > 0)

        ft = self.sag(new_x, new_y, valid) - new_o[..., 2]
        dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]
        dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y, valid)
        dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
        t = t - torch.clamp(
            ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
        )

        # 3. Determine valid solutions
        with torch.no_grad():
            # Solution within the surface boundary. Ray is allowed to go back
            new_x, new_y = new_o[..., 0], new_o[..., 1]
            valid = self.is_valid(new_x, new_y) & (ray.is_valid > 0)

            # Solution accurate enough
            ft = self.sag(new_x, new_y, valid) - new_o[..., 2]
            valid = valid & (torch.abs(ft) < newton_convergence)

        return t, valid

    def refract(self, ray, eta):
        """Calculate refracted ray according to Snell's law in local coordinate system.

        Normal vector points from the surface toward the side where the light is coming from. d is already normalized if both n and ray.d are normalized.

        Args:
            ray (Ray): incident ray.
            eta (float): ratio of indices of refraction, eta = n_i / n_t

        Returns:
            ray (Ray): refracted ray.

        References:
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute normal vectors
        normal_vec = self.normal_vec(ray)

        # Compute refraction according to Snell's law, normal_vec * ray_d
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2) 

        # Total internal reflection
        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        # Update ray direction and obliquity
        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        # ==> Update obliq term to penalize steep rays in the later optimization.
        obliq = torch.sum(new_d * ray.d, axis=-1).unsqueeze(-1)
        obliq_update_mask = valid.unsqueeze(-1) & (obliq < 0.5)
        ray.obliq = torch.where(obliq_update_mask, obliq * ray.obliq, ray.obliq)
        # ==> 
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        # Update ray valid mask
        ray.is_valid = ray.is_valid * valid

        return ray

    def reflect(self, ray):
        """Calculate reflected ray in local coordinate system.

        Normal vector points from the surface toward the side where the light is coming from.

        Args:
            ray (Ray): incident ray.

        Returns:
            ray (Ray): reflected ray.

        References:
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute surface normal vectors
        normal_vec = self.normal_vec(ray)

        # Reflect
        dot_product = (normal_vec * ray.d).sum(-1).unsqueeze(-1)
        new_d = ray.d - 2 * dot_product * normal_vec
        new_d = F.normalize(new_d, p=2, dim=-1)

        # Update valid rays
        valid_mask = ray.is_valid > 0
        ray.d = torch.where(valid_mask.unsqueeze(-1), new_d, ray.d)

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at the intersection point in local coordinate system.

        Normal vector points from the surface toward the side where the light is coming from.

        Args:
            ray (Ray): input ray.

        Returns:
            n_vec (tensor): surface normal vector.
        """
        x, y = ray.o[..., 0], ray.o[..., 1]
        nx, ny, nz = self.dfdxyz(x, y)
        n_vec = torch.stack((nx, ny, nz), axis=-1)
        n_vec = F.normalize(n_vec, p=2, dim=-1)

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        n_vec = torch.where(is_forward, n_vec, -n_vec)
        return n_vec

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
        """Calculate rotation matrix to rotate vec_from to vec_to.

        Args:
            vec_from (tensor): source direction vector [3]
            vec_to (tensor): target direction vector [3]

        Returns:
            R (tensor): rotation matrix [3, 3]
        """
        # CRITICAL: Normalize input vectors
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        # Check if vectors are already aligned
        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            # Vectors are already aligned, return identity matrix
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            # Vectors are opposite, need 180-degree rotation
            # Find a perpendicular vector
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # Get rotation axis by cross product
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)

            # 180-degree rotation matrix
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        # General case: use Rodrigues' rotation formula
        # For normalized vectors: v × u = sin(θ) * k (where k is unit rotation axis)
        # and v · u = cos(θ)
        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        # Skew-symmetric matrix for cross product v × u (not normalized axis!)
        K = torch.tensor(
            [
                [0, -v_cross_u[2], v_cross_u[1]],
                [v_cross_u[2], 0, -v_cross_u[0]],
                [-v_cross_u[1], v_cross_u[0], 0],
            ],
            device=self.device,
        )

        # Rodrigues' formula: R = I + K + K²/(1 + cos(θ))
        # This is equivalent to: R = I + sin(θ)K + (1-cos(θ))K²
        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """Apply rotation matrix to vectors.

        Args:
            vectors (tensor): input vectors [..., 3]
            R (tensor): rotation matrix [3, 3]

        Returns:
            rotated_vectors (tensor): rotated vectors [..., 3]
        """
        original_shape = vectors.shape
        # Reshape to [..., 3] for matrix multiplication
        vectors_flat = vectors.view(-1, 3)
        # Apply rotation: v' = R @ v (transpose for batch operation)
        rotated_flat = torch.mm(vectors_flat, R.t())
        # Reshape back to original shape
        return rotated_flat.view(original_shape)

    # =====================================================================
    # Computation functions
    # =====================================================================
    def sag(self, x, y, valid=None):
        """Calculate sag (z) of the surface: z = f(x, y).

        Valid term is used to avoid NaN when x, y exceed the data range, which happens in spherical and aspherical surfaces.

        Calculating r = sqrt(x**2, y**2) may cause an NaN error during back-propagation. Because dr/dx = x / sqrt(x**2 + y**2), NaN will occur when x=y=0.
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        return self._sag(x, y)

    def _sag(self, x, y):
        """Calculate sag (z) of the surface: z = f(x, y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        raise NotImplementedError(
            "_sag() is not implemented for {}".format(self.__class__.__name__)
        )

    def dfdxyz(self, x, y, valid=None):
        """Compute derivatives of surface function. Surface function: f(x, y, z): sag(x, y) - z = 0. This function is used in Newton's method and normal vector calculation.

        There are several methods to compute derivatives of surfaces:
            [1] Analytical derivatives: The current implementation is based on this method. But the implementation only works for surfaces which can be written as z = sag(x, y). For implicit surfaces, we need to compute derivatives (df/dx, df/dy, df/dz).
            [2] Numerical derivatives: Use finite difference method to compute derivatives. This can be used for those very complex surfaces, for example, NURBS. But it may suffer from numerical instability when the surface is very steep.
            [3] Automatic differentiation: Use torch.autograd to compute derivatives. This can work for almost all the surfaces and is accurate, but it requires an extra backward pass to compute the derivatives of the surface function.
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        dx, dy = self._dfdxy(x, y)
        return dx, dy, -torch.ones_like(x)

    def _dfdxy(self, x, y):
        """Compute derivatives of sag to x and y. (dfdx, dfdy, dfdz) =  (f'x, f'y, f'z).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            dfdx (tensor): df / dx
            dfdy (tensor): df / dy
        """
        raise NotImplementedError(
            "_dfdxy() is not implemented for {}".format(self.__class__.__name__)
        )

    def d2fdxyz2(self, x, y, valid=None):
        """Compute second-order partial derivatives of the surface function f(x, y, z): sag(x, y) - z = 0. This function is currently only used for surfaces constraints."""
        if valid is None:
            valid = self.is_within_data_range(x, y)

        x, y = x * valid, y * valid

        # Compute second-order derivatives of sag(x, y)
        d2f_dx2, d2f_dxdy, d2f_dy2 = self._d2fdxy(x, y)

        # Mixed partial derivatives involving z are zero
        zeros = torch.zeros_like(x)
        d2f_dxdz = zeros  # ∂²f/∂x∂z = 0
        d2f_dydz = zeros  # ∂²f/∂y∂z = 0
        d2f_dz2 = zeros  # ∂²f/∂z² = 0

        return d2f_dx2, d2f_dxdy, d2f_dy2, d2f_dxdz, d2f_dydz, d2f_dz2

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of sag to x and y. (d2fdx2, d2fdxdy, d2fdy2) =  (f''xx, f''xy, f''yy).

        Currently, we use finite difference method to compute the second-order derivatives. And the second-order derivatives are only used for surface constraints.

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            d2fdx2 (tensor): d2f / dx2
            d2fdxdy (tensor): d2f / dxdy
            d2fdy2 (tensor): d2f / dy2
        """
        delta_x = 1e-6
        delta_y = 1e-6
        d2fdx2 = (self._dfdxy(x + delta_x, y)[0] - self._dfdxy(x - delta_x, y)[0]) / (
            2 * delta_x
        )
        d2fdy2 = (self._dfdxy(x, y + delta_y)[1] - self._dfdxy(x, y - delta_y)[1]) / (
            2 * delta_y
        )
        d2fdxy = (self._dfdxy(x + delta_x, y)[1] - self._dfdxy(x - delta_x, y)[1]) / (
            2 * delta_x
        )
        return d2fdx2, d2fdxy, d2fdy2

    def is_valid(self, x, y):
        """Valid points within the data range and boundary of the surface."""
        return self.is_within_data_range(x, y) & self.is_within_boundary(x, y)

    def is_within_boundary(self, x, y):
        """Valid points within the boundary of the surface."""
        if self.is_square:
            valid = (torch.abs(x) <= (self.w / 2 + EPSILON)) & (
                torch.abs(y) <= (self.h / 2 + EPSILON)
            )
        else:
            if self.tolerancing:
                r = self.r + self.r_error
            else:
                r = self.r
            valid = (x**2 + y**2).sqrt() <= r

        return valid

    def is_within_data_range(self, x, y):
        """Valid points inside the data region of the sag function."""
        return torch.ones_like(x, dtype=torch.bool)

    def max_height(self):
        """Maximum valid height."""
        return 10e3

    def surface_with_offset(self, x, y, valid_check=True):
        """Calculate z coordinate of the surface at (x, y).

        This function is used in lens setup plotting and lens self-intersection detection.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        if valid_check:
            return self.sag(x, y) + self.d
        else:
            return self._sag(x, y) + self.d

    def surface_sag(self, x, y):
        """Calculate sag of the surface at (x, y).

        This function is currently not used.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        return self.sag(x, y).item()

    # =====================================================================
    # Optimization
    # =====================================================================

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Get optimizer parameters for different parameters.

        Args:
            lrs (list): learning rates for different parameters.
            optim_mat (bool): whether to optimize material. Defaults to False.
        """
        raise NotImplementedError(
            "get_optimizer_params() is not implemented for {}".format(
                self.__class__.__name__
            )
        )

    def get_optimizer(self, lrs=[1e-4], optim_mat=False):
        """Get optimizer for the surface."""
        params = self.get_optimizer_params(lrs, optim_mat=optim_mat)
        return torch.optim.Adam(params)

    def update_r(self, r):
        """Update surface radius."""
        r_max = self.max_height()
        self.r = min(r, r_max)

    # =====================================================================
    # Tolerancing
    # =====================================================================
    def init_tolerance(self, tolerance_params=None):
        """Initialize tolerance parameters for the surface.

        Args:
            tolerance_params (dict): Tolerance for surface parameters. Example:
                {
                    "r_tole": 0.05, # [mm]
                    "d_tole": 0.05, # [mm]
                    "center_thickness_tole": 0.1, # [mm]
                    "decenter_tole": 0.1, # [mm]
                    "tilt_tole": 0.1, # [arcmin]
                    "mat2_n_tole": 0.001,
                    "mat2_V_tole": 0.01, # [%]
                }

        References:
            [1] https://www.edmundoptics.com/knowledge-center/application-notes/optics/understanding-optical-specifications/?srsltid=AfmBOorBa-0zaOcOhdQpUjmytthZc07oFlmPW_2AgaiNHHQwobcAzWII
            [2] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/08/8-Tolerancing-1.pdf
            [3] https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/L17_OPTI517_Lens-_Tolerancing.pdf
        """
        if tolerance_params is None:
            tolerance_params = {}

        self.r_tole = tolerance_params.get("r_tole", 0.05)
        self.d_tole = tolerance_params.get("d_tole", 0.05)
        self.center_thick_tole = tolerance_params.get("center_thick_tole", 0.1)
        self.decenter_tole = tolerance_params.get("decenter_tole", 0.1)
        self.tilt_tole = tolerance_params.get("tilt_tole", 0.1)
        self.mat2_n_tole = tolerance_params.get("mat2_n_tole", 0.001)
        self.mat2_V_tole = tolerance_params.get("mat2_V_tole", 0.01)

    @torch.no_grad()
    def sample_tolerance(self):
        """Sample one example manufacturing error for the surface."""
        self.r_error = float(np.random.uniform(-self.r_tole, 0))  # [mm]
        self.d_error = float(np.random.randn() * self.d_tole)  # [mm]
        self.center_thick_error = float(np.random.randn() * self.center_thick_tole)
        self.decenter_error = float(np.random.randn() * self.decenter_tole)  # [mm]
        self.tilt_error = float(np.random.randn() * self.tilt_tole)  # [arcmin]
        self.tilt_error = self.tilt_error / 60.0 * np.pi / 180.0  # [rad]
        self.mat2_n_error = float(np.random.randn() * self.mat2_n_tole)
        self.mat2_V_error = float(np.random.randn() * self.mat2_V_tole) * self.mat2.V
        self.tolerancing = True

    def zero_tolerance(self):
        """Zero tolerance."""
        self.r_error = 0.0
        self.d_error = 0.0
        self.center_thick_error = 0.0
        self.decenter_error = 0.0
        self.tilt_error = 0.0
        self.mat2_n_error = 0.0
        self.mat2_V_error = 0.0
        self.tolerancing = False

    def sensitivity_score(self):
        """Tolerance squared sum.

        Reference:
            [1] Page 10 from: https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/08/8-Tolerancing-1.pdf
        """
        score_dict = {}
        score_dict.update(
            {
                f"surf{self.surf_idx}_d_grad": round(self.d.grad.item(), 6),
                f"surf{self.surf_idx}_d_score": round(
                    (self.d_tole**2 * self.d.grad**2).item(), 6
                ),
            }
        )
        return score_dict

    # =====================================================================
    # Visualization
    # =====================================================================
    def draw_widget(self, ax, color="black", linestyle="solid"):
        """Draw widget for the surface on the 2D plot."""
        r = torch.linspace(-self.r, self.r, 128, device=self.device)
        z = self.surface_with_offset(r, torch.zeros(len(r), device=self.device))
        ax.plot(
            z.cpu().detach().numpy(),
            r.cpu().detach().numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=0.75,
        )

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.06, 0.3, 0.6]):
        """Create triangulated surface mesh.

        Args:
            n_rings (int): Number of concentric rings for sampling.
            n_arms (int): Number of angular divisions.
            color (List[float]): The color of the mesh.

        Returns:
            self: The surface with mesh data.
        """
        self.vertices = self._create_vertices(n_rings, n_arms)
        self.faces = self._create_faces(n_rings, n_arms)
        self.rim = self._create_rim(n_rings, n_arms)
        self.mesh_color = color
        return self

    def _create_vertices(self, n_rings, n_arms):
        """Create vertices in radial pattern. Vertices will be used to plot the surface in PyVista."""
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)

        # Center vertex
        vertices[0] = [0.0, 0.0, self.surface_with_offset(0.0, 0.0).item()]

        # Create meshgrid and flatten
        rings_mesh, arms_mesh = np.meshgrid(
            np.linspace(1, self.r, n_rings, endpoint=False),
            np.linspace(0, 2 * np.pi, n_arms, endpoint=False),
            indexing="ij",
        )
        rings_flat = rings_mesh.flatten()
        arms_flat = arms_mesh.flatten()

        # Calculate x, y, z coordinates
        x_values = rings_flat * np.cos(arms_flat)
        y_values = rings_flat * np.sin(arms_flat)
        z_values = self.surface_with_offset(x_values, y_values).cpu().numpy()

        # Fill vertices array
        vertices[1:, 0] = x_values
        vertices[1:, 1] = y_values
        vertices[1:, 2] = z_values

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """Create triangular faces. Faces will be used to plot the surface in PyVista."""
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)
        normal_direction = -1 if self.mat2.name != "air" else 1

        # Create central triangles
        for j in range(n_arms):
            if normal_direction == 1:
                faces[j] = [0, 1 + j, 1 + (j + 1) % n_arms]
            else:
                # Flip winding order for opposite normal direction
                faces[j] = [0, 1 + (j + 1) % n_arms, 1 + j]

        # Create radial quads (2 triangles each)
        face_idx = n_arms

        for i_ring in range(1, n_rings):
            for j_arm in range(n_arms):
                # Get indices for current ring vertices
                a = 1 + (i_ring - 1) * n_arms + j_arm
                b = 1 + (i_ring - 1) * n_arms + (j_arm + 1) % n_arms

                # Get indices for next ring
                c = 1 + i_ring * n_arms + j_arm
                d = 1 + i_ring * n_arms + (j_arm + 1) % n_arms

                # Create two triangles per quad
                if normal_direction == 1:
                    faces[face_idx] = [a, c, b]
                    faces[face_idx + 1] = [b, c, d]
                else:
                    # Flip winding order for opposite normal direction
                    faces[face_idx] = [a, b, c]
                    faces[face_idx + 1] = [b, d, c]
                face_idx += 2

        return faces

    def _create_rim(self, n_rings, n_arms):
        """Create rim (outer edge) vertices. Rims will be used to bridge two surfaces."""
        if n_rings == 0:
            return RimCurve(self.vertices[[0]], is_loop=False)

        # Get outer ring vertices
        start_idx = 1 + (n_rings - 1) * n_arms
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    def get_polydata(self):
        """Get PyVista PolyData object from previously generated vertices and faces.

        PolyData object will be used to draw the surface and export as .obj file.
        """
        from pyvista import PolyData

        face_vertex_n = 3  # vertices per triangle
        formatted_faces = np.hstack(
            [
                face_vertex_n * np.ones((self.faces.shape[0], 1), dtype=np.uint32),
                self.faces,
            ]
        )
        return PolyData(self.vertices, formatted_faces)

    # =====================================================================
    # IO
    # =====================================================================
    def surf_dict(self):
        surf_dict = {
            "type": self.__class__.__name__,
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "pos_xy": (round(self.pos_x.item(), 4), round(self.pos_y.item(), 4)),
            "vec_local": (
                round(self.vec_local[0].item(), 4),
                round(self.vec_local[1].item(), 4),
                round(self.vec_local[2].item(), 4),
            ),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        raise NotImplementedError(
            "zmx_str() is not implemented for {}".format(self.__class__.__name__)
        )


class RimCurve:
    """Simple curve class for surface rim, compatible with LineMesh interface."""

    def __init__(self, vertices, is_loop=False):
        self.vertices = (
            vertices.copy() if hasattr(vertices, "copy") else np.array(vertices)
        )
        self.is_loop = is_loop
        self.n_vertices = len(vertices)

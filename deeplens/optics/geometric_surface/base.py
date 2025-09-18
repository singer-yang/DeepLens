"""Base class for geometric surfaces.

Surface can refract, and reflect rays. Some surfaces can also diffract rays according to a local grating approximation.
"""

import numpy as np
import torch
import torch.nn.functional as F

from deeplens.optics.basics import DeepObj
from deeplens.optics.materials import Material

EPSILON = 1e-12  # [float], small value to avoid division by zero


class Surface(DeepObj):
    def __init__(self, r, d, mat2, is_square=False, surf_idx=None, device="cpu"):
        super(Surface, self).__init__()
        self.surf_idx = surf_idx

        # Surface
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        
        # Next material
        self.mat2 = Material(mat2)

        # Surface aperture radius (non-differentiable parameter)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.h = float(r * np.sqrt(2))
            self.w = float(r * np.sqrt(2))

        # Newton method parameters
        self.newton_maxiter = 10  # [int], maximum number of Newton iterations
        self.newton_convergence = 50.0 * 1e-6  # [mm], Newton method solution threshold
        self.newton_step_bound = self.r / 5  # [mm], maximum step size in each Newton iteration


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
        # ray = self.to_local_coord(ray)

        # Intersection
        ray = self.intersect(ray, n1)

        if refraction:
            # Refraction
            ray = self.refract(ray, n1 / n2)
        else:
            # Reflection
            ray = self.reflect(ray)

        # ray = self.to_global_coord(ray)
        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection and update ray position and opl.

        Args:
            ray (Ray): input ray.
            n (float, optional): refractive index. Defaults to 1.0.
        """
        # Solve intersection time t by Newton's method
        t, valid = self.newtons_method(ray)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.valid = ray.valid * valid

        if ray.coherent:
            if t.min() > 100 and torch.get_default_dtype() == torch.float32:
                raise Exception(
                    "Using float32 may cause precision problem at long propagation distance."
                )
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def newtons_method(self, ray):
        """Solve intersection by Newton's method.

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
            d_surf = self.d + self.d_error
        else:
            d_surf = self.d

        # Initial guess of t
        # Note: Sometimes the shape of aspheric surface is too ambornal, 
        # this step will hit the back surface region and cause error.
        t = (d_surf - ray.o[..., 2]) / ray.d[..., 2]

        # 1. Non-differentiable Newton's iterations to find the intersection points
        with torch.no_grad():
            it = 0
            ft = 1e6 * torch.ones_like(ray.o[..., 2])
            while it < newton_maxiter:
                # Converged
                if (torch.abs(ft) < newton_convergence).all():
                    break

                # Newton iteration
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[..., 0], new_o[..., 1]
                valid = self.is_within_data_range(new_x, new_y) & (ray.valid > 0)

                ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
                dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y, valid)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(
                    ft / (dfdt + EPSILON),
                    -newton_step_bound,
                    newton_step_bound,
                )

        # 2. One more Newton iteration (differentiable) to gain gradients
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[..., 0], new_o[..., 1]
        valid = self.is_valid(new_x, new_y) & (ray.valid > 0)

        ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
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
            valid = self.is_valid(new_x, new_y) & (ray.valid > 0)

            # Solution accurate enough
            ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
            valid = valid & (torch.abs(ft) < newton_convergence)

        return t, valid

    def refract(self, ray, n):
        """Calculate refracted ray according to Snell's law.

        Normal vector points from the surface toward the side where the light is coming from.

        Args:
            ray (Ray): incident ray.
            n (float): relevant refraction coefficient, n = n_i / n_t

        Returns:
            ray (Ray): refracted ray.

        References:
            [1] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute normal vectors
        normal_vec = self.normal_vec(ray)

        # Compute refraction according to Snell's law, normal_vec * ray_d
        cosi = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)

        # Total internal reflection. Shape [N] now, maybe broadcasted to [N, 1] in the future.
        valid = (n**2 * (1 - cosi**2) < 1).squeeze(-1) & (ray.valid > 0)

        # Square root term in Snell's law
        sr = torch.sqrt(1 - n**2 * (1 - cosi**2) * valid.unsqueeze(-1) + EPSILON)

        # Update ray direction and obliquity. d is already normalized if both n and ray.d are normalized.
        new_d = n * ray.d + (n * cosi - sr) * normal_vec
        new_obliq = torch.sum(new_d * ray.d, axis=-1).unsqueeze(-1)
        ray.obliq = torch.where(valid.unsqueeze(-1), new_obliq, ray.obliq)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        # Update ray valid mask
        ray.valid = ray.valid * valid

        return ray

    def reflect(self, ray):
        """Calculate reflected ray.

        Normal vector points from the surface toward the side where the light is coming from.

        Args:
            ray (Ray): incident ray.

        Returns:
            ray (Ray): reflected ray.

        References:
            [1] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute surface normal vectors
        normal_vec = self.normal_vec(ray)

        # Reflect
        ray.is_forward = ~ray.is_forward
        cos_alpha = -(normal_vec * ray.d).sum(-1).unsqueeze(-1)
        new_d = ray.d + 2 * cos_alpha * normal_vec
        new_d = F.normalize(new_d, p=2, dim=-1)

        # Update valid rays
        valid_mask = ray.valid > 0
        ray.d = torch.where(valid_mask.unsqueeze(-1), new_d, ray.d)

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at the intersection point.

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
        n_vec = torch.where(ray.is_forward, n_vec, -n_vec)
        return n_vec

    def to_local_coord(self, ray):
        """Transform ray to local coordinate system.

        Args:
            ray (Ray): input ray in global coordinate system.

        Returns:
            ray (Ray): transformed ray in local coordinate system.
        """
        raise NotImplementedError(
            "to_local_coord() is not implemented for {}".format(self.__class__.__name__)
        )

    def to_global_coord(self, ray):
        """Transform ray to global coordinate system.

        Args:
            ray (Ray): input ray in local coordinate system.

        Returns:
            ray (Ray): transformed ray in global coordinate system.
        """
        raise NotImplementedError(
            "to_global_coord() is not implemented for {}".format(
                self.__class__.__name__
            )
        )

    # =====================================================================
    # Computation functions
    # =====================================================================
    def sag(self, x, y, valid=None):
        """Calculate sag (z) of the surface: z = f(x, y). Valid term is used to avoid NaN when x, y are super large, which happens in spherical and aspherical surfaces.

        Notes:
            If you want to calculate r = sqrt(x**2, y**2), this may cause an NaN error during back-propagation when calculating dr/dx = x / sqrt(x**2 + y**2). So be careful for this!"""
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        return self._sag(x, y)

    def _sag(self, x, y):
        """Calculate sag (z) of the surface. z = f(x, y)

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
            [1] Analytical derivatives: This is the function implemented here. But the current implementation only works for surfaces which can be written as z = sag(x, y). For implicit surfaces, we need to compute derivatives (df/dx, df/dy, df/dz).
            [2] Numerical derivatives: Use finite difference method to compute derivatives. This can be used for those very complex surfaces, for example, NURBS. But it may not be accurate when the surface is very steep.
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
            valid = (torch.abs(x) <= (self.w / 2)) & (torch.abs(y) <= (self.h / 2))
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
        if self.tolerancing:
            r = self.r + self.r_error
        else:
            r = self.r
        return r

    # def surface_sample(self, N=1000):
    #     """Sample uniform points on the surface."""
    #     raise Exception("surface_sample() is deprecated.")
    #     r_max = self.r
    #     theta = torch.rand(N) * 2 * torch.pi
    #     r = torch.sqrt(torch.rand(N) * r_max**2)
    #     x2 = r * torch.cos(theta)
    #     y2 = r * torch.sin(theta)
    #     z2 = torch.full_like(x2, self.d.item())
    #     o2 = torch.stack((x2, y2, z2), 1).to(self.device)
    #     return o2

    # def surface(self, x, y):
    #     """Calculate z coordinate of the surface at (x, y) with offset.

    #     This function is used in lens setup plotting.
    #     """
    #     raise Exception("surface() is deprecated. Use surface_with_offset() instead.")
    #     x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
    #     y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
    #     return self.sag(x, y)

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
        self.r = r

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
        score_dict.update({
            f"surf{self.surf_idx}_d_grad": round(self.d.grad.item(), 6),
            f"surf{self.surf_idx}_d_score": round((self.d_tole**2 * self.d.grad**2).item(), 6),
        })
        return score_dict

    # =====================================================================
    # Visualization
    # =====================================================================
    def draw_widget(self, ax, color="black", linestyle="solid"):
        """Draw wedge for the surface on the plot."""
        r = torch.linspace(-self.r, self.r, 128, device=self.device)
        z = self.surface_with_offset(r, torch.zeros(len(r), device=self.device))
        ax.plot(
            z.cpu().detach().numpy(),
            r.cpu().detach().numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=0.75,
        )

    # def draw_widget3D(self, ax, color="lightblue", res=128):
    #     """Draw the surface in a 3D plot."""
    #     raise Exception("draw_widget3D() is deprecated.")
    #     if self.is_square:
    #         x = torch.linspace(-self.r, self.r, res, device=self.device)
    #         y = torch.linspace(-self.r, self.r, res, device=self.device)
    #         X, Y = torch.meshgrid(x, y, indexing="ij")
    #     else:
    #         r_coords = torch.linspace(0, self.r, res // 2, device=self.device)
    #         theta_coords = torch.linspace(0, 2 * torch.pi, res, device=self.device)
    #         R, Theta = torch.meshgrid(r_coords, theta_coords, indexing="ij")
    #         X = R * torch.cos(Theta)
    #         Y = R * torch.sin(Theta)

    #     # Calculate z coordinates
    #     Z = self.surface_with_offset(X, Y, valid_check=False)

    #     # Convert to numpy for plotting
    #     X_np = X.cpu().detach().numpy()
    #     Y_np = Y.cpu().detach().numpy()
    #     Z_np = Z.cpu().detach().numpy()

    #     # Plot the surface
    #     surf = ax.plot_surface(
    #         Z_np,
    #         X_np,
    #         Y_np,
    #         alpha=0.5,
    #         color=color,
    #         edgecolor="none",
    #         rcount=res,
    #         ccount=res,
    #         antialiased=True,
    #     )

    #     # Draw the edge
    #     if self.is_square:
    #         # Draw square edge
    #         w_half, h_half = self.w / 2, self.h / 2
    #         edge_x_vals = [-w_half, w_half, w_half, -w_half, -w_half]
    #         edge_y_vals = [h_half, h_half, -h_half, -h_half, h_half]
    #         edge_x = []
    #         edge_y = []
    #         edge_z = []
    #         # Sample points along the square edges
    #         for i in range(4):
    #             x_start, x_end = edge_x_vals[i], edge_x_vals[i + 1]
    #             y_start, y_end = edge_y_vals[i], edge_y_vals[i + 1]
    #             num_steps = res // 4
    #             xs = torch.linspace(x_start, x_end, num_steps, device=self.device)
    #             ys = torch.linspace(y_start, y_end, num_steps, device=self.device)
    #             zs = self.surface_with_offset(xs, ys)
    #             edge_x.extend(xs.cpu().numpy())
    #             edge_y.extend(ys.cpu().numpy())
    #             edge_z.extend(zs.cpu().numpy())
    #         ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.0, alpha=1.0)
    #     else:
    #         # Draw circular edge
    #         theta = torch.linspace(0, 2 * torch.pi, res, device=self.device)
    #         edge_x = self.r * torch.cos(theta)
    #         edge_y = self.r * torch.sin(theta)
    #         edge_z_tensor = self.surface_with_offset(
    #             edge_x,
    #             edge_y,
    #             valid_check=False,
    #         )
    #         edge_z = edge_z_tensor.cpu().numpy()
    #         ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.0, alpha=1.0)

    #     return surf

    # =====================================================================
    # IO
    # =====================================================================
    def surf_dict(self):
        surf_dict = {
            "idx": self.surf_idx,
            "type": self.__class__.__name__,
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        raise NotImplementedError(
            "zmx_str() is not implemented for {}".format(self.__class__.__name__)
        )

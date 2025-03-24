"""Base class for geometric surfaces.

Surface can refract, and reflect rays. Some surfaces can also diffract rays according to local grating approximation.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..basics import DeepObj
from ..materials import Material

# Newton's method parameters
NEWTONS_MAXITER = 10  # maximum number of Newton iterations
NEWTONS_TOLERANCE_TIGHT = 25e-6  # [mm], Newton method solution threshold
NEWTONS_TOLERANCE_LOOSE = 50e-6  # [mm], Newton method solution threshold
NEWTONS_STEP_BOUND = 5  # [mm], maximum step size in each Newton iteration

EPSILON = 1e-9


class Surface(DeepObj):
    def __init__(self, r, d, mat2, is_square=False, device="cpu"):
        super(Surface, self).__init__()

        # Surface position
        self.d = d if torch.is_tensor(d) else torch.tensor(d)

        # Surface aperture radius (non-differentiable parameter)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.h = float(r * np.sqrt(2))
            self.w = float(r * np.sqrt(2))

        # Next aterial
        self.mat2 = Material(mat2)

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize surface from a dict."""
        raise NotImplementedError(
            f"init_from_dict() is not implemented for {cls.__name__}."
        )

    # =========================================
    # Intersection, refraction, reflection
    # =========================================
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
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            assert t.min() < 100, (
                "Precision problem caused by long propagation distance."
            )
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def newtons_method(self, ray):
        """Solve intersection by Newton's method.

        Args:
            ray (Ray): input ray.

        Returns:
            t (tensor): intersection time.
            valid (tensor): valid mask.
        """
        if hasattr(self, "d_offset"):
            d_surf = self.d + self.d_offset
        else:
            d_surf = self.d

        # 1. Inital guess of t
        # Note: Sometimes the shape of aspheric surface is too ambornal, this step will hit the back surface region and cause error
        t0 = (d_surf - ray.o[..., 2]) / ray.d[..., 2]

        # 2. Non-differentiable Newton's method to update t and find the intersection points
        with torch.no_grad():
            it = 0
            t = t0  # initial guess of t
            ft = 1e6 * torch.ones_like(ray.o[..., 2])
            while (torch.abs(ft) > NEWTONS_TOLERANCE_LOOSE).any() and (
                it < NEWTONS_MAXITER
            ):
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[..., 0], new_o[..., 1]
                valid = self.is_within_data_range(new_x, new_y) & (ray.ra > 0)

                ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
                dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y, valid)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(
                    ft / (dfdt + 1e-12),
                    -NEWTONS_STEP_BOUND,
                    NEWTONS_STEP_BOUND,
                )

            t1 = t - t0

        # 3. One more Newton iteration (differentiable) to gain gradients
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[..., 0], new_o[..., 1]
        valid = self.is_valid(new_x, new_y) & (ray.ra > 0)

        ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
        dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]
        dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y, valid)
        dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
        t = t - torch.clamp(ft / (dfdt + 1e-9), -NEWTONS_STEP_BOUND, NEWTONS_STEP_BOUND)

        # 4. Determine valid solutions
        with torch.no_grad():
            # Solution within the surface boundary and ray doesn't go back
            new_x, new_y = new_o[..., 0], new_o[..., 1]
            valid = self.is_valid(new_x, new_y) & (ray.ra > 0) & (t >= 0)

            # Solution accurate enough
            ft = self.sag(new_x, new_y, valid) + d_surf - new_o[..., 2]
            valid = valid & (torch.abs(ft) < NEWTONS_TOLERANCE_TIGHT)

        return t, valid

    def refract(self, ray, n):
        """Calculate refractive ray according to Snell's law.

        Snell's law (surface normal n defined along the positive z axis):
            [1] https://physics.stackexchange.com/a/436252/104805
            [2] https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
            We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it points to left. We use the second link to check.

        Args:
            ray (Ray): input ray.
            n (float): relevant refraction coefficient, n = n_i / n_t

        Returns:
            ray (Ray): refractive ray.
        """
        # Compute normal vectors
        n_vec = self.normal_vec(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0
        if forward:
            n_vec = -n_vec

        # Compute refraction according to Snell's law
        cosi = torch.sum(ray.d * n_vec, axis=-1)  # n * i

        # Total internal reflection
        valid = (n**2 * (1 - cosi**2) < 1) & (ray.ra > 0)

        sr = torch.sqrt(
            1 - n**2 * (1 - cosi.unsqueeze(-1) ** 2) * valid.unsqueeze(-1) + EPSILON
        )  # square root

        # First term: vertical. Second term: parallel. Already normalized if both n and ray.d are normalized.
        new_d = sr * n_vec + n * (ray.d - cosi.unsqueeze(-1) * n_vec)

        # Update ray direction
        new_d[~valid] = ray.d[~valid]
        ray.d = new_d

        # Update ray obliquity
        new_obliq = torch.sum(new_d * ray.d, axis=-1)
        new_obliq[~valid] = ray.obliq[~valid]
        ray.obliq = new_obliq

        # Update ray ra
        ray.ra = ray.ra * valid

        return ray

    def reflect(self, ray):
        """Calculate reflected ray.

        Args:
            ray (Ray): input ray.

        Returns:
            ray (Ray): reflected ray.
        """
        # Compute surface normal vectors
        n = self.normal_vec(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0
        if forward:
            n = -n

        # Reflect
        cos_alpha = -(n * ray.d).sum(-1)
        new_d = ray.d + 2 * cos_alpha.unsqueeze(-1) * n
        new_d = F.normalize(new_d, p=2, dim=-1)

        # Update valid rays
        valid = ray.ra > 0
        new_d[~valid] = ray.d[~valid]
        ray.d = new_d

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at the intersection point. Normal vector points to the left by default.

        Args:
            ray (Ray): input ray.

        Returns:
            n_vec (tensor): surface normal vector.
        """
        x, y = ray.o[..., 0], ray.o[..., 1]
        nx, ny, nz = self.dfdxyz(x, y)
        n_vec = torch.stack((nx, ny, nz), axis=-1)

        n_vec = F.normalize(n_vec, p=2, dim=-1)
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

    # =================================================================================
    # Computation functions
    # =================================================================================
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

        Notes:
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
        """Compute second-order derivatives of sag to x and y. (d2gdx2, d2gdxdy, d2gdy2) =  (g''xx, g''xy, g''yy).

        As the second-order derivatives are not commonly used in the lens design, we just return zeros.

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            d2fdx2 (tensor): d2f / dx2
            d2fdxdy (tensor): d2f / dxdy
            d2fdy2 (tensor): d2f / dy2
        """
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    def is_valid(self, x, y):
        """Valid points within the data range and boundary of the surface."""
        return self.is_within_data_range(x, y) & self.is_within_boundary(x, y)

    def is_within_boundary(self, x, y):
        """Valid points within the boundary of the surface.

        NOTE: DELTA is used to avoid the numerical error.
        """
        if self.is_square:
            valid = (torch.abs(x) <= (self.w / 2)) & (torch.abs(y) <= (self.h / 2))
        else:
            valid = (x**2 + y**2).sqrt() <= self.r

        return valid

    def is_within_data_range(self, x, y):
        """Valid points inside the data region of the sag function."""
        return torch.ones_like(x, dtype=torch.bool)

    def surface_sample(self, N=1000):
        """Sample uniform points on the surface."""
        raise Exception("surface_sample() is deprecated.")
        r_max = self.r
        theta = torch.rand(N) * 2 * torch.pi
        r = torch.sqrt(torch.rand(N) * r_max**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, self.d.item())
        o2 = torch.stack((x2, y2, z2), 1).to(self.device)
        return o2

    def surface(self, x, y):
        """Calculate z coordinate of the surface at (x, y) with offset.

        This function is used in lens setup plotting.
        """
        raise Exception("surface() is deprecated. Use surface_with_offset() instead.")
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        return self.sag(x, y)

    def surface_with_offset(self, x, y):
        """Calculate z coordinate of the surface at (x, y) with offset.

        This function is used in lens setup plotting.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        return self.sag(x, y) + self.d

    def max_height(self):
        """Maximum valid height."""
        return self.r

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lr, optim_mat=False):
        raise NotImplementedError(
            "get_optimizer_params() is not implemented for {}".format(
                self.__class__.__name__
            )
        )

    def get_optimizer(self, lr, optim_mat=False):
        params = self.get_optimizer_params(lr, optim_mat=optim_mat)
        return torch.optim.Adam(params)

    # =========================================
    # Manufacturing
    # =========================================
    @torch.no_grad()
    def perturb(self, tolerance):
        """Randomly perturb surface parameters to simulate manufacturing errors.

        Reference:
            [1] Surface precision +0.000/-0.010 mm is regarded as high quality by Edmund Optics.
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/understanding-optical-specifications/?srsltid=AfmBOorBa-0zaOcOhdQpUjmytthZc07oFlmPW_2AgaiNHHQwobcAzWII

        Args:
            tolerance (dict): Tolerance for surface parameters.
        """
        self.r_offset = float(
            self.r * torch.randn(1).item() * tolerance.get("r", 0.001)
        )
        self.d_offset = float(torch.randn(1).item() * tolerance.get("d", 0.001))

    # =========================================
    # Visualization
    # =========================================
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

    def draw_widget3D(self, ax, color="lightblue"):
        """Draw the surface in a 3D plot."""
        resolution = 128

        # Create a grid of x, y coordinates
        x = torch.linspace(-self.r, self.r, resolution, device=self.device)
        y = torch.linspace(-self.r, self.r, resolution, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Calculate z coordinates (surface height)
        valid = self.is_within_boundary(X, Y)
        Z = self.surface_with_offset(X, Y)

        # Convert to numpy for plotting
        X_np = X.cpu().detach().numpy()
        Y_np = Y.cpu().detach().numpy()
        Z_np = Z.cpu().detach().numpy()

        # Mask out invalid points
        mask = valid.cpu().detach().numpy()
        X_np = np.where(mask, X_np, np.nan)
        Y_np = np.where(mask, Y_np, np.nan)
        Z_np = np.where(mask, Z_np, np.nan)

        # Plot the surface
        surf = ax.plot_surface(
            Z_np,
            X_np,
            Y_np,
            alpha=0.5,
            color=color,
            edgecolor="none",
            rcount=resolution,
            ccount=resolution,
            antialiased=True,
        )

        # Draw the edge circle
        theta = np.linspace(0, 2 * np.pi, 100)
        edge_x = self.r * np.cos(theta)
        edge_y = self.r * np.sin(theta)
        edge_z = np.array(
            [
                self.surface_with_offset(
                    torch.tensor(edge_x[i], device=self.device),
                    torch.tensor(edge_y[i], device=self.device),
                ).item()
                for i in range(len(theta))
            ]
        )

        ax.plot(edge_z, edge_x, edge_y, color="lightblue", linewidth=1.0, alpha=1.0)

        return surf

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        surf_dict = {
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

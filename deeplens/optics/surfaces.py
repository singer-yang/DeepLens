"""Refractive and reflective optical surfaces. Use ray tracing to compute intersection and refraction/reflection."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .basics import DELTA, EPSILON, DeepObj
from .materials import Material

# Newton's method parameters
NEWTONS_MAXITER = 10  # maximum number of Newton iterations
NEWTONS_TOLERANCE_TIGHT = 25e-6  # [mm], Newton method solution threshold
NEWTONS_TOLERANCE_LOOSE = 50e-6  # [mm], Newton method solution threshold
NEWTONS_STEP_BOUND = 5  # [mm], maximum step size in each Newton iteration


class Surface(DeepObj):
    def __init__(self, r, d, mat2, is_square=False, device="cpu"):
        super(Surface, self).__init__()

        raise Exception("Surfaces is deprecated. Use deeplens.optics.surface instead.")

        # Surface position
        self.d = d if torch.is_tensor(d) else torch.tensor(d)

        # Surface aperture radius (non-differentiable parameter)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.h = float(r * np.sqrt(2))
            self.w = float(r * np.sqrt(2))

        # Material
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

        # 1. Inital guess of t. (If the shape of aspheric surface is too ambornal, this step will hit the back surface region and cause error)
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
            valid = (torch.abs(x) <= (self.w / 2)) & (
                torch.abs(y) <= (self.h / 2)
            )
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
    # Optimization-related functions
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
    # Tolerance-related functions
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
    # IO and visualization functions
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


class Aperture(Surface):
    def __init__(self, r, d, diffraction=False, device="cpu"):
        """Aperture surface."""
        Surface.__init__(self, r, d, mat2="air", is_square=False, device=device)
        self.diffraction = diffraction
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "diffraction" in surf_dict:
            diffraction = surf_dict["diffraction"]
        else:
            diffraction = False
        return cls(surf_dict["r"], surf_dict["d"], diffraction)

    def ray_reaction(self, ray, n1=1.0, n2=1.0, refraction=False):
        """Compute output ray after intersection and refraction."""
        # Intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) <= self.r) & (
            ray.ra > 0
        )

        # Update position
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        # Update phase
        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        # Diffraction
        if self.diffraction:
            raise Exception("Diffraction is not implemented for aperture.")

        return ray

    def _sag(self, x, y):
        """Compute surface height (always zero for aperture)."""
        return torch.zeros_like(x)

    def draw_widget(self, ax, color="orange", linestyle="solid"):
        """Draw aperture wedge on the figure."""
        d = self.d.item()
        aper_wedge_l = 0.05 * self.r  # [mm]
        aper_wedge_h = 0.15 * self.r  # [mm]

        # Parallel edges
        z = np.linspace(d - aper_wedge_l, d + aper_wedge_l, 3)
        x = -self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

        # Vertical edges
        z = d * np.ones(3)
        x = np.linspace(self.r, self.r + aper_wedge_h, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = np.linspace(-self.r - aper_wedge_h, -self.r, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

    def draw_widget3D(self, ax, color="black"):
        """Draw the aperture as a circle in a 3D plot."""
        # Draw the edge circle
        theta = np.linspace(0, 2 * np.pi, 100)
        edge_x = self.r * np.cos(theta)
        edge_y = self.r * np.sin(theta)
        edge_z = np.full_like(edge_x, self.d.item())  # Constant z at aperture position

        # Plot the edge circle
        line = ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.5)

        return line

    def surf_dict(self):
        """Dict of surface parameters."""
        surf_dict = {
            "type": "Aperture",
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
            "is_square": self.is_square,
            "diffraction": self.diffraction,
        }
        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Zemax surface string."""
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
"""
        return zmx_str


class Aspheric(Surface):
    """Aspheric surface.

    Reference:
        [1] https://en.wikipedia.org/wiki/Aspheric_lens.
    """

    def __init__(self, r, d, c=0.0, k=0.0, ai=None, mat2=None, device="cpu"):
        """Initialize aspheric surface.

        Args:
            r (float): radius of the surface
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): aspheric coefficients
            mat2 (Material): material of the second medium
            device (torch.device): device to store the tensor
        """
        Surface.__init__(self, r, d, mat2, is_square=False, device=device)
        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        if ai is not None:
            self.ai = torch.tensor(ai)
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
            elif self.ai_degree == 5:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
                self.ai10 = torch.tensor(ai[4])
            elif self.ai_degree == 6:
                self.ai2 = torch.tensor(ai[0])
                self.ai4 = torch.tensor(ai[1])
                self.ai6 = torch.tensor(ai[2])
                self.ai8 = torch.tensor(ai[3])
                self.ai10 = torch.tensor(ai[4])
                self.ai12 = torch.tensor(ai[5])
            else:
                for i, a in enumerate(ai):
                    exec(f"self.ai{2 * i + 2} = torch.tensor({a})")
        else:
            self.ai = None
            self.ai_degree = 0

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        if "ai" in surf_dict:
            ai = surf_dict["ai"]
        else:
            ai = torch.rand(6) * 1e-30

        return cls(
            surf_dict["r"], surf_dict["d"], c, surf_dict["k"], ai, surf_dict["mat2"]
        )

    def _sag(self, x, y):
        """Compute surface height."""
        r2 = x**2 + y**2
        total_surface = (
            r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON))
        )

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                )
            elif self.ai_degree == 5:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                    + self.ai10 * r2**5
                )
            elif self.ai_degree == 6:
                total_surface = (
                    total_surface
                    + self.ai2 * r2
                    + self.ai4 * r2**2
                    + self.ai6 * r2**3
                    + self.ai8 * r2**4
                    + self.ai10 * r2**5
                    + self.ai12 * r2**6
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"total_surface += self.ai{2 * i} * r2 ** {i}")

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order height derivatives to x and y."""
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON)
        dsdr2 = (
            (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf) ** 2
        )

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                )
            elif self.ai_degree == 5:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                )
            elif self.ai_degree == 6:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                    + 6 * self.ai12 * r2**5
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"dsdr2 += {i} * self.ai{2 * i} * r2 ** {i - 1}")

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of surface height with respect to x and y."""
        r2 = x**2 + y**2
        c = self.c
        k = self.k
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)

        # Compute dsdr2
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / (2 * sf)) * c / (1 + sf) ** 2

        # Compute derivative of dsdr2 with respect to r2 (ddsdr2_dr2)
        ddsdr2_dr2 = (
            ((1 + k) * c**2 / (2 * sf)) + ((1 + k) ** 2 * r2 * c**4) / (4 * sf**3)
        ) * c / (1 + sf) ** 2 - 2 * dsdr2 * (
            1 + sf + (1 + k) * r2 * c**2 / (2 * sf)
        ) / (1 + sf)

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                )
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
                )
            elif self.ai_degree == 5:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                )
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
                    + 20 * self.ai10 * r2**3
                )
            elif self.ai_degree == 6:
                dsdr2 = (
                    dsdr2
                    + self.ai2
                    + 2 * self.ai4 * r2
                    + 3 * self.ai6 * r2**2
                    + 4 * self.ai8 * r2**3
                    + 5 * self.ai10 * r2**4
                    + 6 * self.ai12 * r2**5
                )
                ddsdr2_dr2 = (
                    ddsdr2_dr2
                    + 2 * self.ai4
                    + 6 * self.ai6 * r2
                    + 12 * self.ai8 * r2**2
                    + 20 * self.ai10 * r2**3
                    + 30 * self.ai12 * r2**4
                )
            else:
                for i in range(1, self.ai_degree + 1):
                    ai_coeff = getattr(self, f"ai{2 * i}")
                    dsdr2 += i * ai_coeff * r2 ** (i - 1)
                    if i > 1:
                        ddsdr2_dr2 += i * (i - 1) * ai_coeff * r2 ** (i - 2)
        else:
            ddsdr2_dr2 = ddsdr2_dr2

        # Compute second-order derivatives
        d2f_dx2 = 2 * dsdr2 + 4 * x**2 * ddsdr2_dr2
        d2f_dxdy = 4 * x * y * ddsdr2_dr2
        d2f_dy2 = 2 * dsdr2 + 4 * y**2 * ddsdr2_dr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        if self.k > -1:
            valid = (x**2 + y**2) < 1 / self.c**2 / (1 + self.k)
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """Maximum valid height."""
        if self.k > -1:
            max_height = torch.sqrt(1 / (self.k + 1) / (self.c**2)).item() - 0.01
        else:
            max_height = 100

        return max_height

    def get_optimizer_params(
        self, lr=[1e-4, 1e-4, 1e-1, 1e-2], decay=0.01, optim_mat=False
    ):
        """Get optimizer parameters for different parameters.

        Args:
            lr (list, optional): learning rates for c, d, k, ai. Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr * 1e3, lr]

        params = []
        if lr[0] > 0 and self.c != 0:
            self.c.requires_grad_(True)
            params.append({"params": [self.c], "lr": lr[0]})

        if lr[1] > 0:
            self.d.requires_grad_(True)
            params.append({"params": [self.d], "lr": lr[1]})

        if lr[2] > 0 and self.k != 0:
            self.k.requires_grad_(True)
            params.append({"params": [self.k], "lr": lr[2]})

        if lr[3] > 0:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
            elif self.ai_degree == 5:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
                params.append({"params": [self.ai10], "lr": lr[3] * decay**4})
            elif self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                params.append({"params": [self.ai2], "lr": lr[3]})
                params.append({"params": [self.ai4], "lr": lr[3] * decay})
                params.append({"params": [self.ai6], "lr": lr[3] * decay**2})
                params.append({"params": [self.ai8], "lr": lr[3] * decay**3})
                params.append({"params": [self.ai10], "lr": lr[3] * decay**4})
                params.append({"params": [self.ai12], "lr": lr[3] * decay**5})
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f"self.ai{2 * i}.requires_grad_(True)")
                    exec(
                        f"params.append({{'params': [self.ai{2 * i}], 'lr': lr[3] * decay**{i - 1}}})"
                    )

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    @torch.no_grad()
    def perturb(self, tolerance):
        """Randomly perturb surface parameters to simulate manufacturing errors.

        Args:
            tolerance (dict): Tolerance for surface parameters.
        """
        self.r_offset = float(self.r * np.random.randn() * tolerance.get("r", 0.001))
        self.c_offset = float(self.c * np.random.randn() * tolerance.get("c", 0.001))
        self.d_offset = float(np.random.randn() * tolerance.get("d", 0.001))
        self.k_offset = float(np.random.randn() * tolerance.get("k", 0.001))
        for i in range(1, self.ai_degree + 1):
            exec(
                f"self.ai{2 * i}_offset = float(np.random.randn() * tolerance.get('ai{2 * i}', 0.001))"
            )

    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": "Aspheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(1 / self.c.item(), 4),
            "d": round(self.d.item(), 4),
            "k": round(self.k.item(), 4),
            "ai": [],
            "mat2": self.mat2.get_name(),
        }
        for i in range(1, self.ai_degree + 1):
            exec(
                f"surf_dict['(ai{2 * i})'] = float(format(self.ai{2 * i}.item(), '.6e'))"
            )
            surf_dict["ai"].append(float(format(eval(f"self.ai{2 * i}.item()"), ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        return zmx_str


class Cubic(Surface):
    """Cubic surface: z(x,y) = b3 * (x**3 + y**3)"""

    def __init__(self, r, d, b, mat2, is_square=False, device="cpu"):
        Surface.__init__(self, r, d, mat2, is_square=is_square, device=device)
        self.b = torch.tensor(b)

        if len(b) == 1:
            self.b3 = torch.tensor(b[0])
            self.b_degree = 1
        elif len(b) == 2:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b_degree = 2
        elif len(b) == 3:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b7 = torch.tensor(b[2])
            self.b_degree = 3
        else:
            raise ValueError("Unsupported cubic degree!")

        self.rotate_angle = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["b"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surface height z(x, y)."""
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = (
                self.b3 * (x**3 + y**3)
                + self.b5 * (x**5 + y**5)
                + self.b7 * (x**7 + y**7)
            )
        else:
            raise ValueError("Unsupported cubic degree!")

        if len(z.size()) == 0:
            z = torch.tensor(z).to(self.device)

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return z

    def _dfdxy(self, x, y):
        """Compute surface height derivatives to x and y."""
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise ValueError("Unsupported cubic degree!")

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return sx, sy

    def get_optimizer_params(self, lr, optim_mat=False):
        """Return parameters for optimizer."""
        params = []

        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lr})

        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
            params.append({"params": [self.b5], "lr": lr * 0.1})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lr})
            params.append({"params": [self.b5], "lr": lr * 0.1})
            params.append({"params": [self.b7], "lr": lr * 0.01})
        else:
            raise ValueError("Unsupported cubic degree!")

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    def perturb(self, tolerance):
        """Perturb the surface"""
        self.r_offset = np.random.randn() * tolerance.get("r", 0.001)
        if self.d != 0:
            self.d_offset = np.random.randn() * tolerance.get("d", 0.0005)

        if self.b_degree == 1:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
        elif self.b_degree == 2:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
            self.b5_offset = np.random.randn() * tolerance.get("b5", 0.001)
        elif self.b_degree == 3:
            self.b3_offset = np.random.randn() * tolerance.get("b3", 0.001)
            self.b5_offset = np.random.randn() * tolerance.get("b5", 0.001)
            self.b7_offset = np.random.randn() * tolerance.get("b7", 0.001)

        self.rotate_angle = np.random.randn() * tolerance.get("angle", 0.01)

    def surf_dict(self):
        """Return surface parameters."""
        return {
            "type": "Cubic",
            "b3": self.b3.item(),
            "b5": self.b5.item(),
            "b7": self.b7.item(),
            "r": self.r,
            "(d)": round(self.d.item(), 4),
        }


class Diffractive_GEO(Surface):
    """Diffractive surfaces simulated with ray tracing.

    Reference:
        https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
    """

    def __init__(
        self, l, d, glass="test", param_model="binary2", thickness=0.5, device="cpu"
    ):
        Surface.__init__(
            self, l / np.sqrt(2), d, mat2="air", is_square=True, device=device
        )

        # DOE geometry
        self.w, self.h = l, l
        self.r = l / float(np.sqrt(2))
        self.l = l
        self.thickness = thickness
        self.glass = glass

        # Use ray tracing to simulate diffraction, the same as Zemax
        self.diffraction = False
        self.diffraction_order = 1
        print(
            "A Diffractive_GEO is created, but ray-tracing diffraction is not activated."
        )

        self.to(device)
        self.init_param_model(param_model)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "glass" in surf_dict:
            glass = surf_dict["glass"]
        else:
            glass = "test"
        if "thickness" in surf_dict:
            thickness = surf_dict["thickness"]
        else:
            thickness = 0.5
        return cls(
            surf_dict["l"], surf_dict["d"], glass, surf_dict["param_model"], thickness
        )

    def init_param_model(self, param_model="binary2"):
        self.param_model = param_model
        if self.param_model == "fresnel":
            # Focal length at 550nm
            self.f0 = torch.tensor(100.0)

        elif self.param_model == "binary2":
            # Zemax binary2 surface type
            self.order2 = torch.tensor(0.0)
            self.order4 = torch.tensor(0.0)
            self.order6 = torch.tensor(0.0)
            self.order8 = torch.tensor(0.0)

        elif self.param_model == "poly1d":
            rand_value = np.random.rand(6) * 0.001
            self.order2 = torch.tensor(rand_value[0])
            self.order3 = torch.tensor(rand_value[1])
            self.order4 = torch.tensor(rand_value[2])
            self.order5 = torch.tensor(rand_value[3])
            self.order6 = torch.tensor(rand_value[4])
            self.order7 = torch.tensor(rand_value[5])

        elif self.param_model == "grating":
            # A grating surface
            self.theta = torch.tensor(0.0)  # angle from x-axis to grating vector
            self.alpha = torch.tensor(0.0)  # slope of the grating

        else:
            raise ValueError(f"Unsupported parameter model: {self.param_model}")

        self.to(self.device)

    def activate_diffraction(self, diffraction_order=1):
        self.diffraction = True
        self.diffraction_order = diffraction_order
        print("Diffraction of DOE in ray tracing is enabled.")

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def ray_reaction(self, ray, n1=None, n2=None):
        """Ray reaction on DOE surface. Imagine the DOE as a wrapped positive convex lens for debugging.

        1, The phase φ in radians adds to the optical path length of the ray
        2, The gradient of the phase profile (phase slope) change the direction of rays.

        https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0

        # Intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        # valid = (new_o[...,0].abs() <= self.h/2) & (new_o[...,1].abs() <= self.w/2) & (ray.ra > 0) # square
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) <= self.r) & (
            ray.ra > 0
        )  # circular
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            # OPL change
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        if self.diffraction:
            # Diffraction 1: DOE phase modulation
            if ray.coherent:
                phi = self.phi(ray.o[..., 0], ray.o[..., 1])
                new_opl = ray.opl + phi * (ray.wvln * 1e-3) / (2 * np.pi)
                new_opl[~valid] = ray.opl[~valid]
                ray.opl = new_opl

            # Diffraction 2: bend rays
            # Perpendicular incident rays are diffracted following (1) grating equation and (2) local grating approximation
            dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

            if forward:
                new_d_x = (
                    ray.d[..., 0]
                    + (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                )
                new_d_y = (
                    ray.d[..., 1]
                    + (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order
                )
            else:
                new_d_x = (
                    ray.d[..., 0]
                    - (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                )
                new_d_y = (
                    ray.d[..., 1]
                    - (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order
                )

            new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
            new_d = F.normalize(new_d, p=2, dim=-1)

            new_d[~valid] = ray.d[~valid]
            ray.d = new_d

        return ray

    def phi(self, x, y):
        """Reference phase map at design wavelength (independent to wavelength). We have the same definition of phase (phi) as Zemax."""
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        if self.param_model == "fresnel":
            phi = (
                -2 * torch.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1)
            )  # unit [mm]

        elif self.param_model == "binary2":
            phi = (
                self.order2 * r**2
                + self.order4 * r**4
                + self.order6 * r**6
                + self.order8 * r**8
            )

        elif self.param_model == "poly1d":
            phi_even = self.order2 * r**2 + self.order4 * r**4 + self.order6 * r**6
            phi_odd = (
                self.order3 * (x_norm**3 + y_norm**3)
                + self.order5 * (x_norm**5 + y_norm**5)
                + self.order7 * (x_norm**7 + y_norm**7)
            )
            phi = phi_even + phi_odd

        elif self.param_model == "grating":
            phi = self.alpha * (
                x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
            )

        else:
            raise NotImplementedError(
                f"phi() is not implemented for {self.param_model}"
            )

        phi = torch.remainder(phi, 2 * np.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points."""
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        if self.param_model == "fresnel":
            dphidx = -2 * np.pi * x / (0.55e-3 * self.f0)  # unit [mm]
            dphidy = -2 * np.pi * y / (0.55e-3 * self.f0)

        elif self.param_model == "binary2":
            dphidr = (
                2 * self.order2 * r
                + 4 * self.order4 * r**3
                + 6 * self.order6 * r**5
                + 8 * self.order8 * r**7
            )
            dphidx = dphidr * x_norm / r / self.r
            dphidy = dphidr * y_norm / r / self.r

        elif self.param_model == "poly1d":
            dphi_even_dr = (
                2 * self.order2 * r + 4 * self.order4 * r**3 + 6 * self.order6 * r**5
            )
            dphi_even_dz = dphi_even_dr * x_norm / r / self.r
            dphi_even_dy = dphi_even_dr * y_norm / r / self.r

            dphi_odd_dx = (
                3 * self.order3 * x_norm**2
                + 5 * self.order5 * x_norm**4
                + 7 * self.order7 * x_norm**6
            ) / self.r
            dphi_odd_dy = (
                3 * self.order3 * y_norm**2
                + 5 * self.order5 * y_norm**4
                + 7 * self.order7 * y_norm**6
            ) / self.r

            dphidx = dphi_even_dz + dphi_odd_dx
            dphidy = dphi_even_dy + dphi_odd_dy

        elif self.param_model == "grating":
            dphidx = self.alpha * torch.sin(self.theta) / self.r
            dphidy = self.alpha * torch.cos(self.theta) / self.r

        else:
            raise NotImplementedError(
                f"dphi_dxy() is not implemented for {self.param_model}"
            )

        return dphidx, dphidy

    def _sag(self, x, y):
        raise ValueError(
            "sag() function is meaningless for phase DOE, use phi() function."
        )

    def _dfdxy(self, x, y):
        raise ValueError(
            "_dfdxy() function is meaningless for DOE, use dphi_dxy() function."
        )

    # def surface(self, x, y, max_offset=0.2):
    #     """When drawing the lens setup, this function is called to compute the surface height.

    #     Here we use a fake height ONLY for drawing.
    #     """
    #     roc = self.l
    #     r = torch.sqrt(x**2 + y**2 + EPSILON)
    #     sag = roc * (1 - torch.sqrt(1 - r**2 / roc**2))
    #     sag = max_offset - torch.fmod(sag, max_offset)
    #     return sag

    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw height map. Range from [0, max_height]."""
        x, y = torch.meshgrid(
            torch.linspace(-self.l / 2, self.l / 2, 2000),
            torch.linspace(self.l / 2, -self.l / 2, 2000),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * np.pi)
        ax[0].set_title("Phase map 0.55um", fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    # ==============================
    # Optimization and other functions
    # ==============================
    def get_optimizer_params(self, lr=None):
        """Generate optimizer parameters."""
        params = []
        if self.param_model == "binary2":
            lr = 0.001 if lr is None else lr
            self.order2.requires_grad = True
            self.order4.requires_grad = True
            self.order6.requires_grad = True
            self.order8.requires_grad = True
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order8], "lr": lr})

        elif self.param_model == "poly1d":
            lr = 0.001 if lr is None else lr
            self.order2.requires_grad = True
            self.order3.requires_grad = True
            self.order4.requires_grad = True
            self.order5.requires_grad = True
            self.order6.requires_grad = True
            self.order7.requires_grad = True
            params.append({"params": [self.order2], "lr": lr})
            params.append({"params": [self.order3], "lr": lr})
            params.append({"params": [self.order4], "lr": lr})
            params.append({"params": [self.order5], "lr": lr})
            params.append({"params": [self.order6], "lr": lr})
            params.append({"params": [self.order7], "lr": lr})

        elif self.param_model == "grating":
            lr = 0.1 if lr is None else lr
            self.theta.requires_grad = True
            self.alpha.requires_grad = True
            params.append({"params": [self.theta], "lr": lr})
            params.append({"params": [self.alpha], "lr": lr})

        else:
            raise NotImplementedError(
                f"get_optimizer_params() is not implemented for {self.param_model}"
            )

        return params

    def get_optimizer(self, lr=None):
        """Generate optimizer.

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-3.
            iterations (float, optional): Iterations. Defaults to 1e4.
        """
        assert self.diffraction, "Diffraction is not activated yet."
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE height map."""
        if self.param_model == "binary2":
            torch.save(
                {
                    "param_model": self.param_model,
                    "order2": self.order2.clone().detach().cpu(),
                    "order4": self.order4.clone().detach().cpu(),
                    "order6": self.order6.clone().detach().cpu(),
                    "order8": self.order8.clone().detach().cpu(),
                },
                save_path,
            )
        elif self.param_model == "poly1d":
            torch.save(
                {
                    "param_model": self.param_model,
                    "order2": self.order2.clone().detach().cpu(),
                    "order3": self.order3.clone().detach().cpu(),
                    "order4": self.order4.clone().detach().cpu(),
                    "order5": self.order5.clone().detach().cpu(),
                    "order6": self.order6.clone().detach().cpu(),
                    "order7": self.order7.clone().detach().cpu(),
                },
                save_path,
            )
        elif self.param_model == "grating":
            torch.save(
                {
                    "param_model": self.param_model,
                    "theta": self.theta.clone().detach().cpu(),
                    "alpha": self.alpha.clone().detach().cpu(),
                },
                save_path,
            )
        else:
            raise ValueError(f"Unknown parameterization: {self.param_model}")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE height map."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        if self.param_model == "binary2" or self.param_model == "poly_even":
            self.param_model = "binary2"
            self.order2 = ckpt["order2"].to(self.device)
            self.order4 = ckpt["order4"].to(self.device)
            self.order6 = ckpt["order6"].to(self.device)
            self.order8 = ckpt["order8"].to(self.device)
        elif self.param_model == "poly1d":
            self.order2 = ckpt["order2"].to(self.device)
            self.order3 = ckpt["order3"].to(self.device)
            self.order4 = ckpt["order4"].to(self.device)
            self.order5 = ckpt["order5"].to(self.device)
            self.order6 = ckpt["order6"].to(self.device)
            self.order7 = ckpt["order7"].to(self.device)
        elif self.param_model == "grating":
            self.theta = ckpt["theta"].to(self.device)
            self.alpha = ckpt["alpha"].to(self.device)
        else:
            raise ValueError(f"Unknown parameterization: {self.param_model}")

    def draw_widget(self, ax, color="black", linestyle="-"):
        """Draw DOE as a two-side surface."""
        max_offset = self.d.item() / 100
        d = self.d.item()

        # Draw DOE
        roc = self.l
        x = np.linspace(-self.r, self.r, 128)
        y = np.zeros_like(x)
        r = np.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - np.sqrt(1 - r**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color=color, linestyle=linestyle, linewidth=0.75)

        # Draw DOE base
        z_bound = [
            d + sag[0],
            d + sag[0] + max_offset,
            d + sag[0] + max_offset,
            d + sag[-1],
        ]
        x_bound = [-self.r, -self.r, self.r, self.r]
        ax.plot(z_bound, x_bound, color=color, linestyle=linestyle, linewidth=0.75)

    def surf_dict(self):
        """Return surface parameters."""
        if self.param_model == "fresnel":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "f0": self.f0.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "binary2":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "order2": self.order2.item(),
                "order4": self.order4.item(),
                "order6": self.order6.item(),
                "order8": self.order8.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "poly1d":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "order2": self.order2.item(),
                "order3": self.order3.item(),
                "order4": self.order4.item(),
                "order5": self.order5.item(),
                "order6": self.order6.item(),
                "order7": self.order7.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        elif self.param_model == "grating":
            surf_dict = {
                "type": self.__class__.__name__,
                "l": self.l,
                "glass": self.glass,
                "param_model": self.param_model,
                "theta": self.theta.item(),
                "alpha": self.alpha.item(),
                "(d)": round(self.d.item(), 4),
                "mat2": self.mat2.get_name(),
            }

        return surf_dict

class Mirror(Surface):
    def __init__(self, l, d, device="cpu"):
        """Mirror surface."""
        Surface.__init__(
            self, l / np.sqrt(2), d, mat2="air", is_square=True, device=device
        )
        self.l = l

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["l"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, **kwargs):
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (
            (torch.abs(new_o[..., 0]) < self.w / 2)
            & (torch.abs(new_o[..., 1]) < self.h / 2)
            & (ray.ra > 0)
        )

        # Update ray position
        new_o = ray.o + ray.d * t.unsqueeze(-1)

        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + 1.0 * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def ray_reaction(self, ray, **kwargs):
        """Compute output ray after intersection and refraction with the mirror surface."""
        # Intersection
        ray = self.intersect(ray)

        # Reflection
        ray = self.reflect(ray)

        return ray

class Plane(Surface):
    def __init__(self, r, d, mat2, is_square=False, device="cpu"):
        """Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base."""
        Surface.__init__(self, r, d, mat2=mat2, is_square=is_square, device=device)
        self.l = r * np.sqrt(2)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection and update ray data."""
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.ra > 0)
            )
        else:
            valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
                ray.ra > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)

        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def normal_vec(self, ray):
        """Calculate surface normal vector at intersection points."""
        n = torch.zeros_like(ray.d)
        n[..., 2] = -1
        return n

    def _sag(self, x, y):
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def _d2fdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    def surf_dict(self):
        surf_dict = {
            "type": "Plane",
            "(l)": self.l,
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "is_square": True,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

class Spheric(Surface):
    """Spheric surface."""

    def __init__(self, c, r, d, mat2, device="cpu"):
        super(Spheric, self).__init__(r, d, mat2, is_square=False, device=device)
        self.c = torch.tensor(c)

        self.c_perturb = 0.0
        self.d_perturb = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]
        return cls(c, surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surfaces sag z = r**2 * c / (1 - sqrt(1 - r**2 * c**2))"""
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt(1 - r2 * c**2))
        return sag

    def _dfdxy(self, x, y):
        """Compute surface sag derivatives to x and y: dz / dx, dz / dy."""
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)
        dfdr2 = c / (2 * sf)

        dfdx = dfdr2 * 2 * x
        dfdy = dfdr2 * 2 * y

        return dfdx, dfdy

    def _d2fdxy(self, x, y):
        """Compute second-order derivatives of the surface sag z = sag(x, y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Returns:
            d2f_dx2 (tensor): ∂²f / ∂x²
            d2f_dxdy (tensor): ∂²f / ∂x∂y
            d2f_dy2 (tensor): ∂²f / ∂y²
        """
        c = self.c + self.c_perturb
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)

        # First derivative (df/dr2)
        dfdr2 = c / (2 * sf)

        # Second derivative (d²f/dr2²)
        d2f_dr2_dr2 = (c**3) / (4 * sf**3)

        # Compute second-order partial derivatives using the chain rule
        d2f_dx2 = 4 * x**2 * d2f_dr2_dr2 + 2 * dfdr2
        d2f_dxdy = 4 * x * y * d2f_dr2_dr2
        d2f_dy2 = 4 * y**2 * d2f_dr2_dr2 + 2 * dfdr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        c = self.c + self.c_perturb

        valid = (x**2 + y**2) < 1 / c**2
        return valid

    def max_height(self):
        """Maximum valid height."""
        c = self.c + self.c_perturb

        max_height = torch.sqrt(1 / c**2).item() - 0.01
        return max_height

    def perturb(self, tolerance):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        self.r_offset = np.random.randn() * tolerance.get("r", 0.001)
        self.d_offset = np.random.randn() * tolerance.get("d", 0.001)

    def get_optimizer_params(self, lr=[0.001, 0.001], optim_mat=False):
        """Activate gradient computation for c and d and return optimizer parameters."""
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.c], "lr": lr[0]})
        params.append({"params": [self.d], "lr": lr[1]})

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    def surf_dict(self):
        """Return surface parameters."""
        roc = 1 / self.c.item() if self.c.item() != 0 else 0.0
        surf_dict = {
            "type": "Spheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(roc, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r} 1 0 0 1 ""
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str


class ThinLens(Surface):
    def __init__(self, f, r, d, mat2="air", is_square=False, device="cpu"):
        """Thin lens surface."""
        Surface.__init__(self, r, d, mat2=mat2, is_square=is_square, device=device)
        self.f = torch.tensor(f)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(surf_dict["f"], surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection and update rays."""
        # Solve intersection
        t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[..., 0] ** 2 + new_o[..., 1] ** 2) < self.r) & (
            ray.ra > 0
        )

        # Update ray position
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray

    def refract(self, ray, n=1.0):
        """For a thin lens, all rays will converge to z = f plane. Therefore we trace the chief-ray (parallel-shift to surface center) to find the final convergence point for each ray.

        For coherent ray tracing, we can think it as a Fresnel lens with infinite refractive index.
        (1) Lens maker's equation
        (2) Spherical lens function
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[..., 2].sum() > 0

        # Calculate convergence point
        if forward:
            t0 = self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(
                xy_final[..., 0].unsqueeze(-1), self.d.item() + self.f.item()
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)
        else:
            t0 = -self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(
                xy_final[..., 0].unsqueeze(-1), self.d.item() - self.f.item()
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)

        # New ray direction
        new_d = o_final - ray.o
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = new_d

        # Optical path length change
        if ray.coherent:
            if forward:
                ray.opl = (
                    ray.opl
                    - (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
            else:
                ray.opl = (
                    ray.opl
                    + (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )

        return ray

    def _sag(self, x, y):
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)

    def draw_widget(self, ax, color="black", linestyle="-"):
        d = self.d.item()
        r = self.r
        ax.annotate(
            "",
            xy=(d, r),
            xytext=(d, -r),
            arrowprops=dict(
                arrowstyle="<->", color=color, linestyle=linestyle, linewidth=0.75
            ),
        )

    def surf_dict(self):
        surf_dict = {
            "type": "ThinLens",
            "f": round(self.f.item(), 4),
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
        }

        return surf_dict

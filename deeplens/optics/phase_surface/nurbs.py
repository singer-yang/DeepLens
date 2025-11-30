"""NURBS (Non-Uniform Rational B-Spline) phase on a plane surface."""

import torch

from deeplens.basics import EPSILON
from deeplens.optics.phase_surface.phase import Phase


class NURBSPhase(Phase):
    """NURBS phase on a plane surface.

    This class implements a diffractive surface where the phase profile is
    represented by a NURBS surface. The NURBS surface is defined by control
    points arranged in a 2D grid, with knot vectors for both u and v directions.

    The surface is evaluated using B-spline basis functions and Cox-de Boor
    recursion algorithm.

    Reference:
        [1] The NURBS Book by Piegl and Tiller
        [2] https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(
        self,
        r,
        d,
        control_points_u=8,
        control_points_v=8,
        degree_u=3,
        degree_v=3,
        control_points=None,
        weights=None,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=True,
        device="cpu",
    ):
        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]
        """Initialize NURBS phase surface.

        Args:
            r: Radius of the surface
            d: Distance to next surface
            control_points_u: Number of control points in u direction (default: 8)
            control_points_v: Number of control points in v direction (default: 8)
            degree_u: Degree of B-spline in u direction (default: 3)
            degree_v: Degree of B-spline in v direction (default: 3)
            control_points: Optional 3D tensor of shape (control_points_u, control_points_v, 3)
                           containing control point coordinates (x, y, z) where z is phase.
                           If None, initialized with small random values.
            weights: Optional 2D tensor of shape (control_points_u, control_points_v)
                    containing weights for rational B-splines. If None, all weights = 1.
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

        # NURBS surface parameters
        self.control_points_u = control_points_u
        self.control_points_v = control_points_v
        self.degree_u = degree_u
        self.degree_v = degree_v

        # Generate knot vectors (clamped B-splines)
        self.knots_u = self._generate_clamped_knots(control_points_u, degree_u)
        self.knots_v = self._generate_clamped_knots(control_points_v, degree_v)

        # Initialize control points (x, y, z) where z represents phase
        if control_points is None:
            # Initialize with small random phase values
            self.control_points = torch.randn(control_points_u, control_points_v, 3, device=device) * 1e-3
            # Set x,y coordinates to be evenly spaced in [-1, 1] range
            u_coords = torch.linspace(0, 1, control_points_u, device=device)
            v_coords = torch.linspace(0, 1, control_points_v, device=device)
            u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
            self.control_points[..., 0] = u_grid * 2 - 1  # x coordinates
            self.control_points[..., 1] = v_grid * 2 - 1  # y coordinates
        else:
            self.control_points = torch.tensor(control_points, dtype=torch.float32, device=device)
            assert self.control_points.shape == (control_points_u, control_points_v, 3), (
                f"control_points must have shape ({control_points_u}, {control_points_v}, 3)"
            )

        # Initialize weights for rational B-splines
        if weights is None:
            self.weights = torch.ones(control_points_u, control_points_v, device=device)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            assert self.weights.shape == (control_points_u, control_points_v), (
                f"weights must have shape ({control_points_u}, {control_points_v})"
            )

        self.to(device)
        self.init_param_model()

    def _generate_clamped_knots(self, n_control_points, degree):
        """Generate clamped knot vector for B-spline.

        Args:
            n_control_points: Number of control points
            degree: B-spline degree

        Returns:
            Knot vector tensor
        """
        n_knots = n_control_points + degree + 1
        knots = torch.zeros(n_knots)

        # Clamped knots: degree+1 zeros at start and end
        knots[:degree+1] = 0.0
        knots[-degree-1:] = 1.0

        # Interior knots evenly spaced
        if n_control_points > degree + 1:
            n_interior = n_control_points - degree - 1
            for i in range(1, n_interior + 1):
                knots[degree + i] = i / (n_interior + 1)

        return knots

    def _find_knot_span(self, knots, degree, u):
        """Find the knot span for parameter u.

        Args:
            knots: Knot vector
            degree: B-spline degree
            u: Parameter value

        Returns:
            Knot span index
        """
        n = len(knots) - degree - 2  # number of control points - 1

        # Handle boundary cases
        if u <= knots[degree]:
            return degree
        if u >= knots[n + 1]:
            return n

        # Binary search for knot span
        low = degree
        high = n + 1
        mid = (low + high) // 2

        while u < knots[mid] or u >= knots[mid + 1]:
            if u < knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    def _basis_functions(self, knots, degree, u, span):
        """Compute B-spline basis functions using Cox-de Boor recursion.

        This implements the standard Piegl-Tiller algorithm from "The NURBS Book".

        Args:
            knots: Knot vector
            degree: B-spline degree
            u: Parameter value
            span: Knot span index

        Returns:
            Array of basis function values
        """
        N = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        left = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        right = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)

        # Initialize zeroth-degree function
        N[0] = 1.0

        # Compute basis functions using Cox-de Boor recursion
        for j in range(1, degree + 1):
            left[j] = u - knots[span + 1 - j]
            right[j] = knots[span + j] - u
            saved = 0.0

            for r in range(j):
                denom = right[r + 1] + left[j - r]
                if denom != 0:
                    temp = N[r] / denom
                else:
                    temp = 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved

        return N

    def _evaluate_nurbs_surface(self, u, v):
        """Evaluate NURBS surface at parameter values (u, v).

        Args:
            u, v: Parameter values (should be in [0, 1] range)

        Returns:
            Surface point (x, y, z) where z is phase value
        """
        # Clamp parameters to valid range
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        # Find knot spans
        span_u = self._find_knot_span(self.knots_u, self.degree_u, u)
        span_v = self._find_knot_span(self.knots_v, self.degree_v, v)

        # Compute basis functions
        Nu = self._basis_functions(self.knots_u, self.degree_u, u, span_u)
        Nv = self._basis_functions(self.knots_v, self.degree_v, v, span_v)

        # Evaluate surface point
        point = torch.zeros(3, dtype=torch.float32, device=self.device)
        weight_sum = 0.0

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                # Control point index
                cp_i = span_u - self.degree_u + i
                cp_j = span_v - self.degree_v + j

                # Skip if indices are out of bounds
                if cp_i < 0 or cp_i >= self.control_points_u or cp_j < 0 or cp_j >= self.control_points_v:
                    continue

                # B-spline basis function value
                basis = Nu[i] * Nv[j]

                # Weight
                weight = self.weights[cp_i, cp_j] * basis

                # Accumulate weighted control point
                point += weight * self.control_points[cp_i, cp_j]
                weight_sum += weight

        # Divide by weight sum for rational B-splines
        if weight_sum > 0:
            point = point / weight_sum

        return point

    def init_param_model(self):
        """Initialize NURBS parameters."""
        self.param_model = "nurbs"

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize NURBS phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        control_points_u = surf_dict.get("control_points_u", 8)
        control_points_v = surf_dict.get("control_points_v", 8)
        degree_u = surf_dict.get("degree_u", 3)
        degree_v = surf_dict.get("degree_v", 3)

        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            control_points_u=control_points_u,
            control_points_v=control_points_v,
            degree_u=degree_u,
            degree_v=degree_v,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # Load control points and weights
        control_points = surf_dict.get("control_points", None)
        if control_points is not None:
            obj.control_points = torch.tensor(control_points, device=obj.device)

        weights = surf_dict.get("weights", None)
        if weights is not None:
            obj.weights = torch.tensor(weights, device=obj.device)

        return obj

    def phi(self, x, y):
        """Reference phase map at design wavelength using NURBS surface evaluation.

        Args:
            x, y: Coordinate tensors

        Returns:
            Phase values in radians at the specified coordinates
        """
        # Normalize coordinates to [0, 1] range for NURBS parameter space
        x_norm = (x / self.norm_radii + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        y_norm = (y / self.norm_radii + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Flatten for batch processing
        x_flat = x_norm.flatten()
        y_flat = y_norm.flatten()
        batch_size = x_flat.shape[0]

        # Evaluate NURBS surface for each point
        phi_values = []
        for i in range(batch_size):
            point = self._evaluate_nurbs_surface(x_flat[i], y_flat[i])
            phi_values.append(point[2])  # z-coordinate contains phase

        phi = torch.stack(phi_values).reshape(x_norm.shape)

        # Apply circular aperture mask (set phase to 0 outside unit circle)
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        phi = torch.where(mask, torch.zeros_like(phi), phi)

        # Ensure phase is in [0, 2π) range
        phi = torch.remainder(phi, 2 * torch.pi)

        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) using NURBS surface.

        Args:
            x, y: Coordinate tensors

        Returns:
            dphidx, dphidy: Phase derivatives in x and y directions
        """
        # For numerical differentiation, compute phi at slightly offset positions
        eps = 1e-6

        # Compute dphi/dx
        phi_x_plus = self.phi(x + eps, y)
        phi_x_minus = self.phi(x - eps, y)
        dphidx = (phi_x_plus - phi_x_minus) / (2 * eps)

        # Compute dphi/dy
        phi_y_plus = self.phi(x, y + eps)
        phi_y_minus = self.phi(x, y - eps)
        dphidy = (phi_y_plus - phi_y_minus) / (2 * eps)

        # Apply circular mask
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        dphidx = torch.where(mask, torch.zeros_like(dphidx), dphidx)
        dphidy = torch.where(mask, torch.zeros_like(dphidy), dphidy)

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters for NURBS control points."""
        params = []

        # Enable gradients for control points (only z-coordinate for phase)
        self.control_points.requires_grad = True
        params.append({"params": [self.control_points], "lr": lrs[0]})

        # Optionally optimize weights
        if len(lrs) > 1:
            self.weights.requires_grad = True
            params.append({"params": [self.weights], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./nurbs_doe.pth"):
        """Save NURBS DOE parameters."""
        torch.save(
            {
                "param_model": "nurbs",
                "control_points": self.control_points.clone().detach().cpu(),
                "weights": self.weights.clone().detach().cpu(),
                "control_points_u": self.control_points_u,
                "control_points_v": self.control_points_v,
                "degree_u": self.degree_u,
                "degree_v": self.degree_v,
                "knots_u": self.knots_u.clone().detach().cpu(),
                "knots_v": self.knots_v.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./nurbs_doe.pth"):
        """Load NURBS DOE parameters."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.control_points = ckpt["control_points"].to(self.device)
        self.weights = ckpt["weights"].to(self.device)
        self.control_points_u = ckpt["control_points_u"]
        self.control_points_v = ckpt["control_points_v"]
        self.degree_u = ckpt["degree_u"]
        self.degree_v = ckpt["degree_v"]
        self.knots_u = ckpt["knots_u"].to(self.device)
        self.knots_v = ckpt["knots_v"].to(self.device)

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": "Phase",
            "r": self.r,
            "is_square": self.is_square,
            "param_model": "nurbs",
            "control_points": self.control_points.clone().detach().cpu().tolist(),
            "weights": self.weights.clone().detach().cpu().tolist(),
            "control_points_u": self.control_points_u,
            "control_points_v": self.control_points_v,
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u.clone().detach().cpu().tolist(),
            "knots_v": self.knots_v.clone().detach().cpu().tolist(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict

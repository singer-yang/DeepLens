"""Aperture surface."""

import numpy as np

from deeplens.optics.geometric_surface.plane import Plane


class Aperture(Plane):
    def __init__(
        self,
        r,
        d,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Aperture surface."""
        Plane.__init__(
            self,
            r=r,
            d=d,
            mat2="air",
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            is_square=surf_dict["is_square"] if "is_square" in surf_dict else False,
            pos_xy=surf_dict["pos_xy"] if "pos_xy" in surf_dict else [0.0, 0.0],
            vec_local=surf_dict["vec_local"] if "vec_local" in surf_dict else [0.0, 0.0, 1.0],
            device=surf_dict["device"] if "device" in surf_dict else "cpu",
        )

    def ray_reaction(self, ray, n1=1.0, n2=1.0, refraction=False):
        """Compute output ray after intersection and refraction."""
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =======================================
    # Visualization
    # =======================================
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

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.0, 0.0, 0.0]):
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
        """Generate vertices for two-ring aperture (inner and outer rings)."""
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        aperture_z = self.d.item()  # All vertices at aperture position
        inner_radius = self.r
        outer_radius = 1.1 * self.r

        # Generate inner ring vertices (first n_arms vertices)
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = inner_radius * np.cos(theta)
            y = inner_radius * np.sin(theta)
            z = aperture_z

            vertices[j_arm] = [x, y, z]

        # Generate outer ring vertices (second n_arms vertices)
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = outer_radius * np.cos(theta)
            y = outer_radius * np.sin(theta)
            z = aperture_z

            vertices[n_arms + j_arm] = [x, y, z]

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """Generate triangular faces connecting inner and outer rings."""
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)

        # Connect inner ring (indices 0 to n_arms-1) to outer ring (indices n_arms to 2*n_arms-1)
        for j_arm in range(n_arms):
            # Inner ring vertices
            inner_a = j_arm
            inner_b = (j_arm + 1) % n_arms

            # Outer ring vertices (offset by n_arms)
            outer_a = n_arms + j_arm
            outer_b = n_arms + (j_arm + 1) % n_arms

            # Create two triangles per quad (normal direction +z)
            face_idx = j_arm * 2
            faces[face_idx] = [inner_a, outer_a, inner_b]
            faces[face_idx + 1] = [inner_b, outer_a, outer_b]

        return faces

    def _create_rim(self, n_rings, n_arms):
        """Create rim (outer edge) vertices for aperture."""
        # Import RimCurve from base module
        from deeplens.optics.geometric_surface.base import RimCurve

        # Get outer ring vertices (second half of vertices array)
        start_idx = n_arms  # Start of outer ring
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4]):
        """Activate gradient computation for d and return optimizer parameters."""
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})

        return params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Dict of surface parameters."""
        surf_dict = {
            "type": "Aperture",
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
            "is_square": self.is_square,
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

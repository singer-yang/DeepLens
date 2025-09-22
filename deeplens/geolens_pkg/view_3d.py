# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

import os
from typing import List

import numpy as np
import torch
from pyvista import Plotter, PolyData, merge

from deeplens.optics import Ray
from deeplens.optics.basics import DEFAULT_WAVE

# ==========================================================
# Mesh class
# (Surface mesh defined in the corresponding surface class)
# ==========================================================

class CrossPoly:
    def __init__(self):
        pass

    def get_polydata(self) -> PolyData:
        pass

    def get_obj_data(self):
        pass

    def draw(self, plotter: Plotter, color: List[float], opacity: float = 1.0):
        """Draw the mesh to the plotter.

        Args:
            plotter: Plotter
            color: List[float]. The color of the mesh.
            opacity: float. The opacity of the mesh.
        """
        poly = self.get_polydata()
        poly["colors"] = np.vstack([color] * poly.n_points)
        plotter.add_mesh(poly, scalars="colors", rgb=True, opacity=opacity)


class LineMesh(CrossPoly):
    def __init__(self, n_vertices, is_loop=False):
        self.n_vertices = n_vertices
        self.is_loop = is_loop
        self.vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        self.create_data()

    def create_data(self):
        pass

    def chain(self, other):
        if self.is_loop or other.is_loop:
            raise ValueError("One of the lines is a loop.")
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.n_vertices = self.vertices.shape[0]
        return None

    def get_polydata(self):
        n_line = 0 if self.is_loop else -1
        n_line += self.n_vertices
        line = [[2, i, (i + 1) % self.n_vertices] for i in range(n_line)]

        return PolyData(self.vertices, lines=line)


class Curve(LineMesh):
    """A curve mesh with vertices and lines. Currently used for ray meshes."""
    
    def __init__(self, vertices: np.ndarray, is_loop: bool = None):
        n_vertices = vertices.shape[0]
        super().__init__(n_vertices, is_loop)
        self.vertices = vertices


class Circle(LineMesh):
    """A circle mesh with normal direction and radius. The normal direciton is defined right-hand rule. Currently not used."""
    
    def __init__(self, n_vertices, origin, direction, radius):
        self.direction = direction
        self.radius = radius
        self.origin = origin
        super().__init__(n_vertices, is_loop=True)

    def create_data(self):
        # Normalize the direction vector
        direction = np.array(self.direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # Find a vector that is not parallel to the direction
        if np.abs(direction[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Use cross product to get perpendicular vectors
        u = np.cross(direction, v1)
        u = u / np.linalg.norm(u)
        v = np.cross(direction, u)
        v = v / np.linalg.norm(v)

        # Generate points on the circle
        origin = np.array(self.origin, dtype=np.float32)
        for i in range(self.n_vertices):
            angle = 2 * np.pi * i / self.n_vertices
            x = self.radius * (u[0] * np.cos(angle) + v[0] * np.sin(angle))
            y = self.radius * (u[1] * np.cos(angle) + v[1] * np.sin(angle))
            z = self.radius * (u[2] * np.cos(angle) + v[2] * np.sin(angle))
            self.vertices[i] = origin + np.array([x, y, z])


class FaceMesh(CrossPoly):
    """A face mesh with vertices and faces. Currently used for bridge meshes."""

    def __init__(self, n_vertices: int, n_faces: int):
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.vertices, self.faces = self._create_empty_data()
        self.rim: LineMesh = None
        self.create_data()
        self.create_rim()

    def _create_empty_data(self):
        vertices = np.zeros((self.n_vertices, 3), dtype=np.float32)
        faces = np.zeros((self.n_faces, 3), dtype=np.uint32)
        return vertices, faces

    def create_data(self):
        pass

    def create_rim(self):
        pass

    def get_mesh(self):
        return self.get_polydata()

    def get_polydata(self) -> PolyData:
        face_vertex_n = 3  # 3 vertices per face
        face = np.hstack(
            [face_vertex_n * np.ones((self.n_faces, 1), dtype=np.uint32), self.faces]
        )
        return PolyData(self.vertices, face)


class RectangleMesh(FaceMesh):
    """A rectangle mesh with vertices and faces. Currently used for sensor meshes."""
    
    def __init__(
        self,
        center: np.ndarray,
        direction_w: np.ndarray,
        direction_h: np.ndarray,
        width: float,
        height: float,
    ):
        # Two directions should be orthogonal
        assert np.dot(direction_w, direction_h) == 0, "Invalid directions"
        # width and height should be positive
        assert width > 0 and height > 0, "Invalid width or height"

        self.center = center
        self.direction_w = direction_w / np.linalg.norm(direction_w)
        self.direction_h = direction_h / np.linalg.norm(direction_h)
        self.width = width
        self.height = height
        super().__init__(n_vertices=4, n_faces=2)

    def create_data(self):
        self.vertices[0] = (
            self.center
            - 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[1] = (
            self.center
            + 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[2] = (
            self.center
            + 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )
        self.vertices[3] = (
            self.center
            - 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )

        self.faces[0] = [0, 1, 2]
        self.faces[1] = [0, 2, 3]


# ====================================================
# Mesh utils
# ====================================================

def bridge(
    l_a: LineMesh,
    l_b: LineMesh,
) -> FaceMesh:
    """Bridge two curves with triangulated faces.
    
    Args:
        l_a : np.ndarray, shape (n_a, 3). The first curve.
        l_b : np.ndarray, shape (n_b, 3). The second curve.
    
    Returns:
        face_mesh (FaceMesh): FaceMesh. Triangulated faces.
    """
    # Check if both lines are loops or both are open
    if l_a.is_loop ^ l_b.is_loop:
        raise ValueError("Both lines must be either loops or open curves.")

    # Check if they have the same number of vertices
    if l_a.n_vertices != l_b.n_vertices:
        raise ValueError("Both lines must have the same number of vertices.")

    n = l_a.n_vertices

    # Align the vertices of l_b to l_a
    if l_a.is_loop:
        # Find the closest vertex in l_b to the first vertex of l_a
        distances = np.linalg.norm(l_b.vertices - l_a.vertices[0], axis=1)
        closest_idx = np.argmin(distances)
        # Reorder l_b's vertices to start from the closest index
        reordered_b = np.roll(l_b.vertices, shift=-closest_idx, axis=0)
    else:
        # Check if the start or end of l_b is closer to the start of l_a
        dist_start = np.linalg.norm(l_b.vertices[0] - l_a.vertices[0])
        dist_end = np.linalg.norm(l_b.vertices[-1] - l_a.vertices[0])
        # Reverse l_b's vertices if the end is closer
        if dist_end < dist_start:
            reordered_b = l_b.vertices[::-1]
        else:
            reordered_b = l_b.vertices.copy()

    # Combine the vertices of l_a and the reordered l_b
    vertices = np.vstack([l_a.vertices, reordered_b])

    # Generate the faces
    faces = []
    if l_a.is_loop:
        for i in range(n):
            j = (i + 1) % n
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])
    else:
        for i in range(n - 1):
            j = i + 1
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])

    faces = np.array(faces, dtype=np.uint32)

    # Create the FaceMesh instance
    face_mesh = FaceMesh(n_vertices=vertices.shape[0], n_faces=faces.shape[0])
    face_mesh.vertices = vertices
    face_mesh.faces = faces

    return face_mesh


# ====================================================
# Ray visualization
# ====================================================

def curve_list_to_polydata(meshes: List[Curve]) -> List[PolyData]:
    """Convert a list of Curve objects to a list of PolyData objects.

    Args:
        meshes: List of Curve objects

    Returns:
        List of PolyData objects
    """
    return [c.get_polydata() for c in meshes]


def geolens_ray_poly(
    lens,
    fovs: List[float],
    fov_phis: List[float],
    n_rings: int = 3,
    n_arms: int = 4,
) -> List[List[Curve]]:
    """
    Sample parallel rays to draw the lens setup.\\
    Hx, Hy = fov * cos(fov_phi), fov * sin(fov_phi).\\
    Px, Py are sampled using Zemax like rings & arms method.\\
        
    Args:
        lens: GeoLens. The lens object.
        fovs: List[float]. FoV angles to be sampled, unit: degree.
        fov_phis: List[float]. FoV azimuthal angles to be sampled, unit: degree.
        n_rings: int. Number of pupil rings to be sampled.
        n_arms: int. Number of pupil arms to be sampled.
    
    Returns:
        rays_poly: List[List[Curve]]. Traced ray represented by curves. Each FoV coord is a List[Curve].
        (num_fovs, num_fov_phis, num_rays, 3)
    """
    rays_poly = []

    R = lens.surfaces[0].r

    for fov in fovs:
        if fov == 0.0:
            center_ray = sample_parallel_3D(lens, R, rings=n_rings, arms=n_arms)
            rays_poly.append(curve_from_trace(lens, center_ray))
        else:
            for fov_phi in fov_phis:
                print(f"fov: {fov}, fov_phi: {fov_phi}")
                # Sample rays on the fov
                ray = sample_parallel_3D(
                    lens,
                    R,
                    rings=n_rings,
                    arms=n_arms,
                    view_polar=fov,
                    view_azi=fov_phi,
                )
                rays_poly.append(curve_from_trace(lens, ray))
    return rays_poly


def sample_parallel_3D(
    lens,
    R: float,
    wvln=DEFAULT_WAVE,
    z=None,
    view_polar: float = 0.0,
    view_azi: float = 0.0,
    rings: int = 3,
    arms: int = 4,
    forward: bool = True,
    entrance_pupil=True,
):
    """
    Sample 2D parallel rays. Rays have shape [M, 3].

    Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to infinity

    Args:
        R (float, optional): sampling radius. Defaults to None.
        wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        z (float, optional): sampling depth. Defaults to None.
        view_polar (float, optional): POLAR incident angle (in degree). Defaults to 0.0.
        view_azi (float, optional): AZIMUTHAL incident angle (in degree). Defaults to 0.0.
        rings (int, optional): rings of ray number. Defaults to 5.
        arms (int, optional): arms of ray number. Defaults to 8.
        forward (bool, optional): forward or backward rays. Defaults to True.
        entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
    """
    pass
    if entrance_pupil:
        # Sample 2nd points on the pupil
        pupilz, pupilx = lens.calc_entrance_pupil()
    else:
        pupilz, pupilx = 0, lens.surfaces[0].r

    # x2 = torch.linspace(-pupilx, pupilx, M) * 0.99
    rho2 = torch.linspace(0, pupilx, rings + 1) * 0.99
    rho2 = rho2[1:]  # remove the central spot
    phi2 = torch.linspace(0, 2 * np.pi, arms + 1)
    phi2 = phi2[:-1]
    RHO2, PHI2 = torch.meshgrid(rho2, phi2, indexing="ij")
    X2, Y2 = RHO2 * torch.cos(PHI2), RHO2 * torch.sin(PHI2)
    x2, y2 = torch.flatten(X2), torch.flatten(Y2)

    # add the central spot back
    x2 = torch.concat((torch.tensor([0]), x2))
    y2 = torch.concat((torch.tensor([0]), y2))

    z2 = torch.full_like(x2, pupilz)
    o2 = torch.stack((x2, y2, z2), axis=-1)  # shape [M, 3]

    view_polar = view_polar / 57.3
    view_azi = view_azi / 57.3
    dx = torch.full_like(x2, np.sin(view_polar) * np.cos(view_azi))
    dy = torch.full_like(x2, np.sin(view_polar) * np.sin(view_azi))
    dz = torch.full_like(x2, np.cos(view_polar))
    d = torch.stack((dx, dy, dz), axis=-1)

    # Move ray origins to z = - 0.1 for tracing
    if pupilz > 0:
        o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
    else:
        o = o2

    return Ray(o, d, wvln, device=lens.device)


def curve_from_trace(lens, ray: Ray, delete_vignetting=True):
    """
    Trace the ray and return the Curve.

    ## Parameters
    - lens: GeoLens
        The lens object.
    - ray: Ray
        Sampled ray from the lens.
    - delete_vignetting: bool
        Whether to delete the vignetting rays.

    ## Returns
    - rays_curve: List[Curve]
        Traced ray represented by curves
    """
    ray, ray_o_records = lens.trace2sensor(ray=ray, record=True)
    rays_curve = []
    # the shape of ray_o_records if [n_surf, M, 3] ?
    ray_o_records = torch.stack(ray_o_records, dim=0)
    ray_o_records = ray_o_records.permute(1, 0, 2).cpu().numpy()
    if delete_vignetting:
        # how to handle the vignetting rays?
        # currently all rays with "nan" are passed to poly
        # this need to be fixed
        pass
    for record in ray_o_records:
        curve = Curve(record, False)
        rays_curve.append(curve)
    return rays_curve


# ====================================================
# Mesh visualization
# ====================================================

class GeoLensVis3D:
    """GeoLens utility class for visualizing the lens geometry and rays in 3D."""
    
    @staticmethod
    def draw_mesh(plotter, mesh: CrossPoly, color: List[float], opacity: float = 1.0):
        """Draw a mesh to the plotter.

        Args:
            plotter: Plotter
            mesh: CrossPoly
            color: List[float]. The color of the mesh.
            opacity: float. The opacity of the mesh.
        """
        poly = mesh.get_polydata() # PolyData object
        plotter.add_mesh(poly, color=color, opacity=opacity)


    def create_mesh(
        self,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
    ):
        """Create all lens/bridge/sensor/aperture meshes.

        Args:
            lens (GeoLens): The lens object.
            mesh_rings (int): The number of rings in the mesh.
            mesh_arms (int): The number of arms in the mesh.

        Returns:
            surf_meshes (List[Surface]): Lens surfaces meshes.
            bridge_meshes (List[FaceMesh]): Lens bridges meshes. (NOT support wrap around for now)
            sensor_mesh (RectangleMesh): Sensor meshes. (only support rectangular sensor for now)
        """
        surf_meshes = []
        element_group = []
        element_groups = []
        bridge_meshes = []
        sensor_mesh = None

        # Create the surface meshes 
        for i, surf in enumerate(self.surfaces):
            # Create the surface mesh (list of Surface objects)
            surf_meshes.append(surf.create_mesh(n_rings=mesh_rings, n_arms=mesh_arms))
                
            # Add the surface to the element group
            element_group.append(i)
            if surf.mat2.name == "air":
                element_groups.append(element_group)
                element_group = []
        
        # Create the bridge meshes (list of FaceMesh objects)
        for i, pair in enumerate(element_groups):
            if len(pair) == 1:
                continue
            elif len(pair) == 2:
                a_idx, b_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                bridge_mesh = bridge(a.rim, b.rim)
                bridge_meshes.append(bridge_mesh)
            elif len(pair) == 3:
                a_idx, b_idx, c_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                c = surf_meshes[c_idx]
                bridge_mesh = bridge(a.rim, b.rim)
                bridge_meshes.append(bridge_mesh)
                bridge_mesh = bridge(b.rim, c.rim)
                bridge_meshes.append(bridge_mesh)
            else:
                raise ValueError(f"Invalid bridge group length: {len(pair)}")

        # Create the sensor mesh (RectangleMesh object)
        sensor_d = self.d_sensor.item()
        sensor_r = self.r_sensor
        h, w = sensor_r * 1.4142, sensor_r * 1.4142
        sensor_mesh = RectangleMesh(
            np.array([0, 0, sensor_d]), np.array([1, 0, 0]), np.array([0, 1, 0]), w, h
        )

        return surf_meshes, bridge_meshes, element_groups, sensor_mesh


    def draw_lens_3d(
        self,
        save_dir: str = None,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        surface_color: List[float] = [0.06, 0.3, 0.6],
        draw_rays: bool = True,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
    ):
        """Draw lens 3D layout with rays using pyvista.

        Args:
            lens (GeoLens): The lens object.
            save_dir (str): The directory to save the image.
            mesh_rings (int): The number of rings in the mesh.
            mesh_arms (int): The number of arms in the mesh.
            surface_color (List[float]): The color of the surfaces.
            draw_rays (bool): Whether to show the rays.
            fovs (List[float]): The FoV angles to be sampled, unit: degree.
            fov_phis (List[float]): The FoV azimuthal angles to be sampled, unit: degree.
            ray_rings (int): The number of pupil rings to be sampled.
            ray_arms (int): The number of pupil arms to be sampled.
        """
        surf_color = np.array(surface_color)
        sensor_color = np.array([0.5, 0.5, 0.5])

        # Initialize plotter
        plotter = Plotter(window_size=(3840, 2160), off_screen=True)
        plotter.camera.up = [0, 1, 0]
        unit = self.d_sensor.item()
        plotter.camera.position = [-2 * unit, unit, -unit / 2]
        plotter.camera.focal_point = [0, 0, unit / 2]
        
        # Create meshes
        surf_meshes, bridge_meshes, _, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms
        )

        # Draw meshes
        for surf in surf_meshes:
            self.draw_mesh(plotter, surf, color=surf_color, opacity=0.5)

        for bridge in bridge_meshes:
            self.draw_mesh(plotter, bridge, color=surf_color, opacity=0.5)
        
        self.draw_mesh(plotter, sensor_mesh, color=sensor_color, opacity=1.0)

        # Draw rays
        if draw_rays:
            rays_curve = geolens_ray_poly(self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms)
            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            for r in rays_poly_fov:
                plotter.add_mesh(r)

        # Save images
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plotter.screenshot(os.path.join(save_dir, "lens_layout3d.png"), return_img=False)


    def save_lens_obj(
        self,
        save_dir: str,
        mesh_rings: int = 128,
        mesh_arms: int = 256,
        save_rays: bool = False,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        save_elements: bool = True,
    ):
        """Save lens geometry and rays as .obj files using pyvista.
        
        Args:
            lens (GeoLens): The lens object.
            save_dir (str): The directory to save the image.
            mesh_rings (int): The number of rings in the mesh. (default: 128)
            mesh_arms (int): The number of arms in the mesh. (default: 256)
            save_rays (bool): Whether to save the rays.
            fovs (List[float]): The FoV angles to be sampled, unit: degree.
            fov_phis (List[float]): The FoV azimuthal angles to be sampled, unit: degree.
            ray_rings (int): The number of pupil rings to be sampled. (default: 6)
            ray_arms (int): The number of pupil arms to be sampled. (default: 8)
            save_elements (bool): Whether to save the elements.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create surfaces & bridges meshes
        surf_meshes, bridge_meshes, element_groups, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms
        )

        if save_elements:
            bridge_idx = 0
            for i, pair in enumerate(element_groups):
                if len(pair) == 1:
                    element = surf_meshes[pair[0]].get_polydata()
                    element.save(os.path.join(save_dir, f"element_{i}.obj"))
                elif len(pair) == 2:
                    a_idx, b_idx = pair
                    surf1 = surf_meshes[a_idx].get_polydata()
                    surf2 = surf_meshes[b_idx].get_polydata()
                    bridge_mesh = bridge_meshes[bridge_idx].get_polydata()
                    bridge_idx += 1
                    element = merge([surf1, surf2, bridge_mesh])
                    element.save(os.path.join(save_dir, f"element_{i}.obj"))
                elif len(pair) == 3:
                    a_idx, b_idx, c_idx = pair
                    surf1 = surf_meshes[a_idx].get_polydata()
                    surf2 = surf_meshes[b_idx].get_polydata()
                    surf3 = surf_meshes[c_idx].get_polydata()
                    bridge1 = bridge_meshes[bridge_idx].get_polydata()
                    bridge_idx += 1
                    bridge2 = bridge_meshes[bridge_idx].get_polydata()
                    bridge_idx += 1
                    element = merge([surf1, surf2, surf3, bridge1, bridge2])
                    element.save(os.path.join(save_dir, f"element_{i}.obj"))
                else:
                    raise ValueError(f"Invalid bridge group length: {len(pair)}")

        # Merge all surfaces and bridges, and save as single lens.obj file
        surf_polydata = [surf.get_polydata() for surf in surf_meshes]
        bridge_polydata = [bridge.get_polydata() for bridge in bridge_meshes]
        lens_polydata = surf_polydata + bridge_polydata
        lens_polydata = merge(lens_polydata)
        lens_polydata.save(os.path.join(save_dir, "lens.obj"))    
        
        # Save sensor
        sensor_polydata = sensor_mesh.get_polydata()
        sensor_polydata.save(os.path.join(save_dir, "sensor.obj"))

        # Save rays
        if save_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )
            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            for i, r in enumerate(rays_poly_fov):
                r.save(os.path.join(save_dir, f"lens_rays_fov_{i}.obj"))
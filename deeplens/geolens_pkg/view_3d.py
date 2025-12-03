# Copyright 2025 Ziqing Zhao, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""3D visualization for geometric lens systems.

GeoLensVis3D class:
    - create_mesh(): Create all lens/bridge/sensor/aperture meshes
    - draw_lens_3d(): Draw lens 3D layout with rays using pyvista
    - save_lens_obj(): Save lens geometry and rays as .obj files
"""

import os
from typing import List, Optional

import numpy as np
import torch

from deeplens.basics import DEFAULT_WAVE
from deeplens.optics import Ray
from deeplens.optics.geometric_surface import Aperture


# ==========================================================
# Mesh class
# (Surface mesh defined in the corresponding surface class)
# ==========================================================
# local dummy class for pyvista
class PolyData:
    def __init__(self, vertices, lines, faces):
        self.n_points = len(vertices)
        self.points = vertices
        self.lines = lines
        self.faces = faces
        self.is_linemesh = False
        self.is_facemesh = False
        self.is_default = False
        if lines is not None:
            self.is_linemesh = True
        if faces is not None:
            self.is_facemesh = True

        assert not (self.is_linemesh and self.is_facemesh), "Invalid polydata"

    def save(self, filename: str):
        # the local wrapper of the pyvista.PolyData.save method
        # only support .obj format for now

        with open(filename, "w") as f:
            mesh_head = "l" if self.is_linemesh else "f"
            v_head = "v"
            if self.is_linemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for l in self.lines:
                    f.write(f"{mesh_head} {l[0] + 1} {l[1] + 1}\n")
            if self.is_facemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for fm in self.faces:
                    f.write(f"{mesh_head} {fm[0] + 1} {fm[1] + 1} {fm[2] + 1}\n")

    # IMPLEMENT A DEFAULT METHOD FOR THE DUMMY CLASS
    @staticmethod
    def default():
        """
        Returns a default PolyData instance that can be used for type checks
        and placeholder initialization. The default instance has an `is_default`
        attribute set to True, which can be used to check for default status.
        """
        obj = PolyData(np.zeros((0, 3)), lines=None, faces=None)
        obj.is_default = True
        return obj


def merge(meshes: List[PolyData]) -> PolyData:
    if meshes is None or len(meshes) == 0:
        return PolyData.default()
    if len(meshes) == 1:
        return meshes[0]
    v_count = meshes[0].n_points
    v_combined = meshes[0].points.copy()
    is_linemesh = meshes[0].is_linemesh
    mesh_combined = meshes[0].lines.copy() if is_linemesh else meshes[0].faces.copy()

    for m in meshes[1:]:
        # increment the vertex number by previous v_count
        if m.is_linemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_lines = m.lines.copy()
            new_lines += v_count
            mesh_combined = np.vstack([mesh_combined, new_lines])
        elif m.is_facemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_faces = m.faces.copy()
            new_faces += v_count
            mesh_combined = np.vstack([mesh_combined, new_faces])
        v_count += m.n_points
    return (
        PolyData(v_combined, lines=mesh_combined, faces=None)
        if is_linemesh
        else PolyData(v_combined, lines=None, faces=mesh_combined)
    )


class CrossPoly:
    def __init__(self):
        pass

    def get_polydata(self) -> PolyData:
        return PolyData.default()

    def get_obj_data(self):
        pass


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
        line = np.array(
            [[i, (i + 1) % self.n_vertices] for i in range(n_line)], dtype=np.uint32
        )
        return PolyData(self.vertices, lines=line, faces=None)


class Curve(LineMesh):
    """A curve mesh with vertices and lines. Currently used for ray meshes."""

    def __init__(self, vertices: np.ndarray, is_loop: Optional[bool] = None):
        if is_loop is None:
            is_loop = False
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
        self.rim: LineMesh = None  # type: ignore
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
        return PolyData(self.vertices, lines=None, faces=self.faces)


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


def line_translate(l: LineMesh, dx: float, dy: float, dz: float) -> LineMesh:
    """Translate a line mesh by a given amount. Return a new line mesh.

    Args:
        l: LineMesh. The line mesh to translate.
        dx: float. The amount to translate in x direction.
        dy: float. The amount to translate in y direction.
        dz: float. The amount to translate in z direction.

    Returns:
        LineMesh. The translated line mesh.
    """
    # create a new line mesh
    new_l = LineMesh(l.n_vertices, l.is_loop)
    new_l.vertices = l.vertices.copy()
    new_l.vertices = new_l.vertices + np.array([dx, dy, dz])[None, :]
    return new_l


def surf_to_face_mesh(surf) -> FaceMesh:
    """Convert a Surface object to a FaceMesh object.

    Args:
        surf: Surface. The surface object.

    Returns:
        FaceMesh. The face mesh object.
    """
    n_vertices = surf.vertices.shape[0]
    n_faces = surf.faces.shape[0]
    face_mesh = FaceMesh(n_vertices=n_vertices, n_faces=n_faces)
    face_mesh.vertices = surf.vertices
    face_mesh.faces = surf.faces
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
    o2 = torch.stack((x2, y2, z2), dim=-1)  # shape [M, 3]

    view_polar = view_polar / 57.3
    view_azi = view_azi / 57.3
    dx = torch.full_like(x2, np.sin(view_polar) * np.cos(view_azi))
    dy = torch.full_like(x2, np.sin(view_polar) * np.sin(view_azi))
    dz = torch.full_like(x2, np.cos(view_polar))
    d = torch.stack((dx, dy, dz), dim=-1)

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
# PyVista GUI helpers (lazy-loaded)
# ====================================================


def _wrap_base_poly_to_pyvista(poly: PolyData, pv):
    """
    Wrap the PolyData object to a pyvista.PolyData object.

    Args:
        poly: PolyData
        pv: pyvista module (passed to avoid top-level import)

    Returns:
        pv.PolyData
    """
    if poly.is_default:
        return pv.PolyData()
    else:
        p = poly.points
        m = poly.lines if poly.is_linemesh else poly.faces
        if poly.is_linemesh:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 2 * _add_on
            new_m = np.hstack([_add_on, m])
        else:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 3 * _add_on
            new_m = np.hstack([_add_on, m])
        return (
            pv.PolyData(p, lines=new_m)
            if poly.is_linemesh
            else pv.PolyData(p, faces=new_m)
        )


def _draw_mesh_to_plotter(
    plotter, mesh: CrossPoly, color: List[float], opacity: float, pv
):
    """
    Draw a mesh to the plotter.

    Args:
        plotter: pv.Plotter
        mesh: CrossPoly
        color: List[float]. The color of the mesh.
        opacity: float. The opacity of the mesh.
        pv: pyvista module (passed to avoid top-level import)
    """
    poly = _wrap_base_poly_to_pyvista(mesh.get_polydata(), pv)
    plotter.add_mesh(poly, color=color, opacity=opacity)


# ====================================================
# Mesh visualization
# ====================================================


class GeoLensVis3D:
    """GeoLens utility class for geometry/ray mesh creation and export (no GUI deps)."""

    # # Attribute stubs to satisfy type checkers when mixed into GeoLens
    # surfaces: List[Any]
    # d_sensor: Any
    # r_sensor: float

    def create_mesh(
        self,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        is_wrap: bool = False,
    ):
        """Create all lens/bridge/sensor/aperture meshes.

        Args:
            lens (GeoLens): The lens object.
            mesh_rings (int): The number of rings in the mesh.
            mesh_arms (int): The number of arms in the mesh.
            is_wrap (bool): Whether to wrap the lens bridge around the lens as cylinder.
        Returns:
            surf_meshes (List[Surface]): Lens surfaces meshes.
            bridge_meshes (List[FaceMesh]): Lens bridges meshes. (NOT support wrap around for now)
            sensor_mesh (RectangleMesh): Sensor meshes. (only support rectangular sensor for now)
        """
        surf_meshes = []
        element_group = []
        element_groups = []
        bridge_meshes = []  # change to nested list for wrap around
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
                bridge_meshes.append([])
                continue
            elif len(pair) == 2:
                a_idx, b_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # create wrap by creating a new rim
                    # from projecting the larger rim onto the smaller rim plane
                    # assume the elements are always ordered on z-axis
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # calc rim mean z
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)

                    if r_a > r_b:
                        z = line_translate(a.rim, 0, 0, d_rim_b - d_rim_a)
                        bridge_mesh_wrap = bridge(z, b.rim)
                        bridge_mesh = bridge(a.rim, z)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    elif r_a < r_b:
                        z = line_translate(b.rim, 0, 0, d_rim_a - d_rim_b)
                        bridge_mesh_wrap = bridge(a.rim, z)
                        bridge_mesh = bridge(z, b.rim)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    else:
                        bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            elif len(pair) == 3:
                a_idx, b_idx, c_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                c = surf_meshes[c_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(b.rim, c.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # create wrap by creating a new rim
                    # from projecting the larger rim onto the smaller rim plane
                    # assume the elements are always ordered on z-axis
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    r_c = self.surfaces[c_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # calc rim mean z
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)
                    d_rim_c = np.mean(c.rim.vertices[:, 2], keepdims=False)

                    rim_list = [a.rim, b.rim, c.rim]
                    r_list = [r_a, r_b, r_c]
                    d_rim_list = [d_rim_a, d_rim_b, d_rim_c]
                    idx_wrap = r_list.index(max(r_list))
                    r_wrap = r_list[idx_wrap]
                    d_rim_wrap = d_rim_list[idx_wrap]

                    for i in range(3):
                        if i != idx_wrap and r_list[i] != r_wrap:
                            # substitute the rim with the wrapped rim
                            d_diff = d_rim_list[i] - d_rim_wrap
                            z = line_translate(rim_list[idx_wrap], 0, 0, d_diff)
                            # add the wrap bridge between older rim and wrapped one
                            wrap_mesh = bridge(rim_list[i], z)
                            # update the rim
                            rim_list[i] = z
                            bridge_mesh_group.append(wrap_mesh)
                    bridge_mesh = bridge(rim_list[0], rim_list[1])
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(rim_list[1], rim_list[2])
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            else:
                raise ValueError(f"Invalid bridge group length: {len(pair)}")

        # Create the sensor mesh (RectangleMesh object)
        sensor_d = self.d_sensor.item()
        sensor_r = self.r_sensor
        h, w = sensor_r * 1.4142, sensor_r * 1.4142
        sensor_mesh = RectangleMesh(
            np.array([0, 0, sensor_d]), np.array([1, 0, 0]), np.array([0, 1, 0]), w, h
        )

        # turn surf_meshes to list of FaceMesh
        surf_meshes_cvt = [surf_to_face_mesh(surf) for surf in surf_meshes]
        return surf_meshes_cvt, bridge_meshes, element_groups, sensor_mesh

    def draw_lens_3d(
        self,
        plotter=None,
        save_dir: Optional[str] = None,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        surface_color: List[float] = [0.06, 0.3, 0.6],
        draw_rays: bool = True,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
    ):
        """Draw lens 3D layout with rays using pyvista.

        Note: PyVista is imported lazily only when this method is called.

        Args:
            plotter: pv.Plotter. Optional pyvista Plotter instance. If None, a new one is created.
            save_dir (str): The directory to save the image.
            mesh_rings (int): The number of rings in the mesh.
            mesh_arms (int): The number of arms in the mesh.
            surface_color (List[float]): The color of the surfaces.
            draw_rays (bool): Whether to show the rays.
            fovs (List[float]): The FoV angles to be sampled, unit: degree.
            fov_phis (List[float]): The FoV azimuthal angles to be sampled, unit: degree.
            ray_rings (int): The number of pupil rings to be sampled.
            ray_arms (int): The number of pupil arms to be sampled.
            is_wrap (bool): Whether to wrap the lens bridge around the lens as cylinder.

        Returns:
            plotter: pv.Plotter. The pyvista Plotter instance.
        """
        # Lazy import of pyvista
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "PyVista is required for 3D GUI rendering. Install with `pip install pyvista`."
            ) from e

        # Create plotter if not provided
        if plotter is None:
            plotter = pv.Plotter()

        surf_color = surface_color
        sensor_color = [0.5, 0.5, 0.5]

        # Create meshes
        surf_meshes, bridge_meshes, _, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # Draw meshes
        for surf in surf_meshes:
            if not isinstance(surf, Aperture):
                _draw_mesh_to_plotter(
                    plotter, surf, color=surf_color, opacity=0.5, pv=pv
                )

        for bridge_group in bridge_meshes:
            for bridge_mesh in bridge_group:
                _draw_mesh_to_plotter(
                    plotter, bridge_mesh, color=surf_color, opacity=0.5, pv=pv
                )

        _draw_mesh_to_plotter(
            plotter, sensor_mesh, color=sensor_color, opacity=1.0, pv=pv
        )

        # Draw rays
        if draw_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )

            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            rays_poly_fov = [_wrap_base_poly_to_pyvista(r, pv) for r in rays_poly_fov]
            for r in rays_poly_fov:
                plotter.add_mesh(r)

        # Save images
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plotter.show(screenshot=os.path.join(save_dir, "lens_layout3d.png"))

        return plotter

    def save_lens_obj(
        self,
        save_dir: str,
        mesh_rings: int = 64,
        mesh_arms: int = 128,
        save_rays: bool = False,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
        save_elements: bool = True,
    ):
        """Save lens geometry and rays as .obj files using pyvista.

        Note: use #F2F7FFFF as the color for lens when rendering in Blender.

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
            is_wrap (bool): Whether to wrap the lens bridge around the lens as cylinder.
            save_elements (bool): Whether to save the elements.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create surfaces & bridges meshes
        surf_meshes, bridge_meshes, element_groups, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # Save individual lens elements (surfaces + bridges merged)
        if save_elements:
            for i, pair in enumerate(element_groups):
                print(f"Running in pair {i} with pair length {len(pair)}")
                # Collect surface polydata
                surf_polydata_list = [surf_meshes[idx].get_polydata() for idx in pair]

                # Collect bridge polydata if available
                bridge_polydata_list = []
                if i < len(bridge_meshes) and len(bridge_meshes[i]) > 0:
                    print(f"Bridge mesh group number: {len(bridge_meshes[i])}")
                    bridge_polydata_list = [b.get_polydata() for b in bridge_meshes[i]]

                # Merge surfaces and bridges together
                all_polydata = surf_polydata_list + bridge_polydata_list
                if len(all_polydata) == 1:
                    element = all_polydata[0]
                else:
                    element = merge(all_polydata)
                element.save(os.path.join(save_dir, f"element_{i}.obj"))

        # Merge all surfaces and bridges, and save as single lens.obj file
        surf_polydata = [
            surf.get_polydata()
            for surf in surf_meshes
            if not isinstance(surf, Aperture)
        ]
        bridge_polydata = [
            b.get_polydata() for group in bridge_meshes for b in group
        ]  # flatten the nested list
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

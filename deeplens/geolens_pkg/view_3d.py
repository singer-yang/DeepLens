# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

import numpy as np
from pyvista import PolyData, merge
from math import pi, sqrt
from typing import List
from os import mkdir
from os import path as osp

from deeplens.geolens import GeoLens
from deeplens.optics.basics import (
    DEFAULT_WAVE
)
from deeplens.optics import (
    Ray,
)
from deeplens.optics.geometric_surface import (
    Aperture,
    Aspheric,
    Cubic,
    Phase,
    Plane,
    Spheric,
    ThinLens,
)

from deeplens.optics.geometric_surface.base import EPSILON

import torch

class CrossPoly:
    def __init__(self):
        pass
    def get_poly_data(self) -> PolyData:
        pass
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

    def get_poly_data(self):
        n_line = 0 if self.is_loop else -1
        n_line += self.n_vertices
        line = [[2, i, (i+1)%self.n_vertices] for i in range(n_line)]
        
        return PolyData(self.vertices, lines=line)

class Curve(LineMesh):
    def __init__(self, vertices: np.ndarray, is_loop: bool = None):
        n_vertices = vertices.shape[0]
        super().__init__(n_vertices, is_loop)
        self.vertices = vertices
        
class LineSeg(LineMesh):
    def __init__(self, origin: np.ndarray, direction: np.ndarray, length: float):
        self.origin = origin
        self.direction = direction
        self.length = length
        super().__init__(2, is_loop=False)

    def create_data(self):
        self.vertices[0] = self.origin
        self.vertices[1] = self.origin + self.direction * self.length

class Circle(LineMesh):
    def __init__(self, n_vertices, origin, direction, radius):
        """
        Create a circle mesh with normal direction and radius.\\
        The normal direciton is defined right-hand rule.\\
        
        """
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
    
    def get_poly_data(self) -> PolyData:
        face_vertex_n = 3 #
        face = np.hstack([face_vertex_n*np.ones((self.n_faces, 1), dtype=np.uint32), self.faces])
        return PolyData(self.vertices, face)

class Rectangle(FaceMesh):
    def __init__(self,
                 center: np.ndarray,
                 direction_w: np.ndarray,
                 direction_h: np.ndarray,
                 width: float,
                 height: float):

        # two directions should be orthogonal
        assert np.dot(direction_w, direction_h) == 0, "Invalid directions"
        # width and height should be positive
        assert width > 0 and height > 0, "Invalid width or height"
        
        self.center = center
        self.direction_w = direction_w / np.linalg.norm(direction_w)
        self.direction_h = direction_h / np.linalg.norm(direction_h)
        self.width = width
        self.height = height
        super().__init__(4, 2)
    
    def create_data(self):
        self.vertices[0] = self.center - 0.5*self.width*self.direction_w - 0.5*self.height*self.direction_h
        self.vertices[1] = self.center + 0.5*self.width*self.direction_w - 0.5*self.height*self.direction_h
        self.vertices[2] = self.center + 0.5*self.width*self.direction_w + 0.5*self.height*self.direction_h
        self.vertices[3] = self.center - 0.5*self.width*self.direction_w + 0.5*self.height*self.direction_h
        
        self.faces[0] = [0, 1, 2]
        self.faces[1] = [0, 2, 3]

class ApertureMesh(FaceMesh):
    def __init__(self, origin: np.ndarray,
                 direction: np.ndarray,
                 aperture_radius: float,
                 radius: float,
                 n_vertices: int = 64):
        """
        Define a circular aperture with radius.\\
        The aperture is defined by the center and radius.\\
        ## Parameters
        - origin: np.ndarray, shape (3,)
            The center of the aperture.
        - direction: np.ndarray, shape (3,)
            Normal direction. Right-hand rule.
        - aperture_radius: float
            The radius of the clear aperture.
        - radius: float
            The radius of the aperture outer rim.
        - n_vertices: int
            The number of vertices in one circle.
        """
        self.origin = origin
        self.direction = direction
        self.aperture_radius = aperture_radius
        self.radius = radius
        super().__init__(n_vertices, n_vertices*2)
    
    def create_data(self):
        inner_circ = Circle(self.n_vertices,
                            self.origin,
                            self.direction,
                            self.aperture_radius)
        outer_circ = Circle(self.n_vertices,
                            self.origin,
                            self.direction,
                            self.radius)
        # bridge the two circles
        bridge_mesh = bridge(inner_circ, outer_circ)
        self.vertices = bridge_mesh.vertices
        self.faces = bridge_mesh.faces
        self.rim = outer_circ
        

class HeightMapAngular(FaceMesh):
    """
    Triangulate a height map on a circular base with angular sampling
    """
    def __init__(self,
                 radius: float,
                 n_rings: int,
                 n_arms: int,
                 height_func: callable):
        assert n_rings > 0 and n_arms > 2, "Invalid number of rings or arms"
        assert radius > 0, "Invalid radius"
        assert callable(height_func), "Invalid height function"
        
        self.radius = radius
        self.n_rings = n_rings
        self.n_arms = n_arms
        self.height_func = height_func

        # Calculate correct grid parameters
        n_vertices = n_rings * n_arms + 1  # central + verteces on rings
        n_faces = n_arms * (2 * n_rings - 1)  # central + outer triangle
        
        super().__init__(n_vertices, n_faces)

    def create_data(self):
        # Generate vertices
        self._generate_vertices()
        # Generate faces
        self._generate_faces()

    def _generate_vertices(self):
        # Center vertex
        self.vertices[0] = [0.0, 0.0, self.height_func(0.0, 0.0)]
        
        # Generate ring vertices
        for i_ring in range(1, self.n_rings+1):
            r = self.radius * i_ring / self.n_rings
            
            for j_arm in range(self.n_arms):
                theta = 2 * np.pi * j_arm / self.n_arms
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = self.height_func(x, y)
                
                idx = 1 + (i_ring-1)*self.n_arms + j_arm
                self.vertices[idx] = [x, y, z]

    def _generate_faces(self):
        # Generate central triangles
        for j in range(self.n_arms):
            self.faces[j] = [0, 1+j, 1 + (j+1)%self.n_arms]
        
        # Generate radial quads
        face_idx = self.n_arms  # index start after central
        
        for i_ring in range(1, self.n_rings):
            for j_arm in range(self.n_arms):
                # Get indices for current ring
                a = 1 + (i_ring-1)*self.n_arms + j_arm
                b = 1 + (i_ring-1)*self.n_arms + (j_arm+1) % self.n_arms
                
                # Get indices for next ring
                c = 1 + i_ring*self.n_arms + j_arm
                d = 1 + i_ring*self.n_arms + (j_arm+1) % self.n_arms
                
                # Create two triangles per quad
                self.faces[face_idx] = [a, c, b]
                self.faces[face_idx+1] = [b, c, d]
                face_idx += 2

    def create_rim(self):
        """
        store the outer most verteces as a LineMesh
        """
        # if self.n_rings == 0:
        #     self.rim = np.array([0], dtype=np.uint32)
        # else:
        #     start_idx = 1 + (self.n_rings-1)*self.n_arms
        #     self.rim = start_idx + np.arange(self.n_arms)
        #     # rim is closed
        #     self.rim = np.append(self.rim, start_idx)
        start_idx = 1 + (self.n_rings-1)*self.n_arms
        self.rim = Curve(self.vertices[start_idx:], is_loop=True)

# ====================================================
# Polydata utils
# ====================================================

def bridge(l_a: LineMesh,
            l_b: LineMesh,
            )-> FaceMesh:
    """
    Bridge two curves/loops with triangulated faces.
    ## Parameters
    - l_a : np.ndarray, shape (n_a, 3).
        The first curve/loop.
    - l_b : np.ndarray, shape (n_b, 3).
        The second curve/loop.
    ## Returns
    - FaceMesh. Return the triangulated faces.
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

def curve_list_to_polydata(meshes: List[Curve]) -> List[PolyData]:
    """Convert a list of Curve objects to a list of PolyData objects.
    
    Args:
        meshes: List of Curve objects
        
    Returns:
        List of PolyData objects
    """
    return [c.get_poly_data() for c in meshes]
    
def group_linemesh(meshes: List[PolyData]) -> PolyData:
    """
    Combine multiple line-based PolyData objects into a single PolyData.
    
    This function merges multiple PolyData objects containing lines into one
    cohesive PolyData object while preserving the line connectivity information.
    The vertex indices in the line connectivity arrays are adjusted to account 
    for the combined vertex array.
    
    ## Parameters:
    
    - meshes: List of PolyData objects to be combined
        
    ## Returns:
    
    - A single PolyData object containing all vertices and lines
    """
    if not meshes:
        return None
    
    # Calculate vertex count offsets
    n_vertices = [mesh.n_points for mesh in meshes]
    vertex_offsets = np.cumsum([0] + n_vertices[:-1])
    
    # Combine all vertices
    combined_vertices = np.vstack([mesh.points for mesh in meshes])
    def offset_vertex(x: np.ndarray, offset):
        x = x.reshape(-1, 3)
        x[:, 1:] += offset
        return x
    
    # Combine all lines with adjusted vertices
    combined_lines = np.vstack([ offset_vertex(mesh.lines, vertex_offsets[i]) for i, mesh in enumerate(meshes)])
    return PolyData(combined_vertices, lines=combined_lines)

# ====================================================
# Height map generation
# ====================================================

def gen_sphere_height_map(c: float, d: float):
    if c == 0:
        def height_func(x, y):
            return d
    else:
        r = np.abs(1/c)
        sign = c / np.abs(c)
        def height_func(x, y):
            z = d + sign*(r - np.sqrt(r**2 - x**2 - y**2 + EPSILON))
            return z
    return height_func

def gen_aspheric_height_map(surf: Aspheric):
    def height_func(x, y):
        return surf._sag(x, y).cpu().numpy() + surf.d.item()
    return height_func

def gen_cubic_height_map(surf: Cubic):
    def height_func(x, y):
        return surf._sag(x, y).cpu().numpy() + surf.d.item()
    return height_func

# ====================================================
# Polygon generation & visualization
# ====================================================

def draw_mesh(plotter, mesh: CrossPoly, color):
    poly = mesh.get_poly_data()
    # shade each points
    n_v = poly.n_points
    poly["colors"] = np.vstack([color]*n_v)
    plotter.add_mesh(poly, scalars="colors", rgb=True)
    
def draw_lens_3D(plotter, lens:GeoLens,
                 fovs: List[float] = [0.],
                 fov_phis: List[float] = [0.],
                 ray_rings: int = 6,
                 ray_arms: int = 8,
                 mesh_rings: int = 32,
                 mesh_arms: int = 128,
                 is_show_bridge: bool = True,
                 is_show_aperture: bool = True,
                 is_show_sensor: bool = True,
                 is_show_rays: bool = True,
                 surface_color: List[float] = [0.06, 0.3, 0.6],
                 bridge_color: List[float] = [0., 0., 0.],
                 save_dir: str = None,):
    n_surf = len(lens.surfaces)
    surf_poly, bridge_poly, sensor_poly, ap_poly = geolens_poly(lens,
                                                                mesh_rings,
                                                                mesh_arms)

    surf_color_rgb = np.array(surface_color) * 255
    surf_color_rgb = surf_color_rgb.astype(np.uint8)
    
    # draw the surfaces
    for sp in surf_poly:
        if sp is not None:
            draw_mesh(plotter, sp, surf_color_rgb)

    if is_show_bridge:
        for bp in bridge_poly:
            draw_mesh(plotter, bp, bridge_color)
    if is_show_aperture:
        for ap in ap_poly:
            draw_mesh(plotter, ap, bridge_color)
    if is_show_sensor:
        draw_mesh(plotter, sensor_poly, np.array([10, 10, 10]))

    if is_show_rays:
        rays_curve = geolens_ray_poly(lens, fovs, fov_phis,
                                    n_rings=ray_rings,
                                    n_arms=ray_arms)
        rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
        rays_poly_fov = [merge(r) for r in rays_poly_list]
        for r in rays_poly_fov:
            plotter.add_mesh(r)
        
    if save_dir is not None:
        if not osp.exists(save_dir):
            mkdir(save_dir)
        # merge meshes
        merged_surf_poly = merge([sp.get_poly_data() for sp in surf_poly if sp is not None])
        merged_bridge_poly = merge([bp.get_poly_data() for bp in bridge_poly])
        merged_ap_poly = merge([ap.get_poly_data() for ap in ap_poly])
        merged_sensor_poly = sensor_poly.get_poly_data()
        
        # save meshes
        merged_surf_poly.save(osp.join(save_dir, "lens_surf.obj"))
        merged_bridge_poly.save(osp.join(save_dir, "lens_bridge.obj"))
        merged_ap_poly.save(osp.join(save_dir, "lens_ap.obj"))
        merged_sensor_poly.save(osp.join(save_dir, "lens_sensor.obj"))
        
        # save rays
        for i, r in enumerate(rays_poly_fov):
            r.save(osp.join(save_dir, f"lens_rays_fov_{i}.obj"))        
        
def geolens_poly(lens: GeoLens,
                 mesh_rings: int = 32,
                 mesh_arms: int = 128,
                 ) -> List[CrossPoly]:
    """
    Generate the lens/bridge/sensor/aperture meshes.\\
    The meshes are generated using the height map method.\\
        
    ## Parameters
    - lens: GeoLens
        The lens object.
    - mesh_rings: int
        The number of rings in the mesh.
    - mesh_arms: int
        The number of arms in the mesh.
    
    ## Returns
    - surf_poly: List[HeightMapAngular]
        The surface meshes.
    - bridge_poly: List[FaceMesh]
        The bridge meshes. (NOT support wrap around for now)
    - sensor_poly: Rectangle
        The sensor meshes. (only support rectangular sensor for now)
    - ap_poly: List[ApertureMesh]
    """
    n_surf = len(lens.surfaces)
    
    surf_poly = [None for _ in range(n_surf)]
    bridge_idx = []
    bridge_poly = []
    ap_poly = []
    sensor_poly = None
    
    radius_list = [surf.r for surf in lens.surfaces]
    max_barrel_r = max(radius_list)
    
    for i, surf in enumerate(lens.surfaces):
        if isinstance(surf, Aperture):
            # generate the aperture mesh
            ap_origin = np.array([0, 0, surf.d.item()])
            ap_dir = np.array([0, 0, -1])
            ap_radius = surf.r
            outer_radius = max_barrel_r
            ap_poly.append(ApertureMesh(ap_origin,
                                        ap_dir,
                                        ap_radius,
                                        outer_radius,
                                        n_vertices=32)) 
        elif isinstance(surf, Spheric):
            # record the idx of the two surf
            # NOTICE:
            # this implementation only consider
            # situation where lens is placed in air
            # non-air material adjacent surf are bridged
            if i < n_surf-1 and surf.mat2.name != "air":
                bridge_idx.append([i, i+1])
                
            # create the surf poly
            r = surf.r
            c = surf.c.item() # brutally assume c is a scalar tensor
            d = surf.d.item()
            height_func = gen_sphere_height_map(c, d)
            surf_poly[i] = HeightMapAngular(r, mesh_rings, mesh_arms, height_func)
        elif isinstance(surf, Aspheric):
            if i < n_surf-1 and surf.mat2.name != "air":
                bridge_idx.append([i, i+1])
            height_func = gen_aspheric_height_map(surf)
            surf_poly[i] = HeightMapAngular(surf.r, mesh_rings, mesh_arms, height_func)
        elif isinstance(surf, Cubic):
            if i < n_surf-1 and surf.mat2.name != "air":
                bridge_idx.append([i, i+1])
            height_func = gen_cubic_height_map(surf)
            surf_poly[i] = HeightMapAngular(surf.r, mesh_rings, mesh_arms, height_func)
        else:
            raise NotImplementedError("Surface type not implemented in 3D visualization")
    
    print(f"Finishing creating {n_surf} surfaces")

    for i, pair in enumerate(bridge_idx):
        print(f"bridging pair: {pair} surfaces")
        
        a_idx, b_idx = pair
        a = surf_poly[a_idx]
        b = surf_poly[b_idx]
        # bridge the two surfaces
        bridge_mesh = bridge(a.rim, b.rim)
        bridge_poly.append(bridge_mesh)
    
    
    sensor_d = lens.d_sensor.item()
    sensor_r = lens.r_sensor
    h, w = sensor_r * 1.4142, sensor_r * 1.4142
    sensor_poly = Rectangle(np.array([0, 0, sensor_d]), 
                             np.array([1, 0, 0]),
                             np.array([0, 1, 0]),
                             w, h)
    
    return surf_poly, bridge_poly, sensor_poly, ap_poly

def geolens_ray_poly(lens: GeoLens,
                     fovs: List[float],
                     fov_phis: List[float],
                     n_rings: int = 3,
                     n_arms: int = 4,
                     ) -> List[List[Curve]]:
    """
    Sample parallel rays to draw the lens setup.\\
    Hx, Hy = fov * cos(fov_phi), fov * sin(fov_phi).\\
    Px, Py are sampled using Zemax like rings & arms method.\\
        
    ## Parameters
    - lens: GeoLens
        The lens object.
    - fovs: List[float]
        FoV angles to be sampled, unit: degree.
    - fov_phis: List[float]
        FoV azimuthal angles to be sampled, unit: degree.
    - n_rings: int
        Number of pupil rings to be sampled.
    - n_arms: int
        Number of pupil arms to be sampled.
    
    ## Returns
    - rays_poly: List[List[Curve]]
        Traced ray represented by curves. Each FoV coord is a List[Curve].
    """
    rays_poly = []
    
    R = lens.surfaces[0].r

    for fov in fovs:
        if fov == 0.:
            center_ray = sample_parallel_3D(lens, R, rings=n_rings, arms=n_arms)
            rays_poly.append(curve_from_trace(lens, center_ray))
        else:
            for fov_phi in fov_phis:
                print(f"fov: {fov}, fov_phi: {fov_phi}")
                # Sample rays on the fov
                ray = sample_parallel_3D(lens, R,
                                        rings=n_rings, arms=n_arms,
                                        view_polar=fov, view_azi=fov_phi)
                rays_poly.append(curve_from_trace(lens, ray))
    return rays_poly

def sample_parallel_3D(lens: GeoLens,
                       R: float,
                       wvln=DEFAULT_WAVE,
                       z=None,
                       view_polar: float = 0.,
                       view_azi: float = 0.,
                       rings: int = 3,
                       arms: int = 4,
                       forward: bool = True,
                       entrance_pupil=True):
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
    rho2 = torch.linspace(0, pupilx, rings+1) * 0.99
    rho2 = rho2[1:] # remove the central spot
    phi2 = torch.linspace(0, 2*pi, arms+1)
    phi2 = phi2[:-1]
    RHO2, PHI2 = torch.meshgrid(rho2, phi2)
    X2, Y2 = RHO2*torch.cos(PHI2), RHO2*torch.sin(PHI2)
    x2, y2 = torch.flatten(X2), torch.flatten(Y2)
    
    # add the central spot back
    x2 = torch.concat((torch.tensor([0]), x2))
    y2 = torch.concat((torch.tensor([0]), y2))
    
    z2 = torch.full_like(x2, pupilz)
    o2 = torch.stack((x2, y2, z2), axis=-1)  # shape [M, 3]

    view_polar = view_polar / 57.3
    view_azi = view_azi / 57.3
    dx = torch.full_like(x2, np.sin(view_polar)*np.cos(view_azi))
    dy = torch.full_like(x2, np.sin(view_polar)*np.sin(view_azi))
    dz = torch.full_like(x2, np.cos(view_polar))
    d = torch.stack((dx, dy, dz), axis=-1)

    # Move ray origins to z = - 0.1 for tracing
    if pupilz > 0:
        o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
    else:
        o = o2

    return Ray(o, d, wvln, device=lens.device)


def curve_from_trace(lens: GeoLens, ray: Ray, delete_vignetting=True):
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
    n_surf = lens.surfaces.__len__()
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
# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# Optional PyVista-based GUI visualization kept separate from core geometry/export.

from typing import List, Optional
import os
import numpy as np
from deeplens import GeoLens

# Import only non-GUI helpers from the core module
from .view_3d import (
    Curve,
    geolens_ray_poly,
    CrossPoly,
    Aperture,
    merge
)
from .view_3d import PolyData as BasePolyData

try:
    import pyvista as pv  # type: ignore
except Exception as e:
    raise ImportError(
        "PyVista is required for 3D GUI rendering. Install with `pip install pyvista`."
    ) from e

def _wrap_base_poly(poly: BasePolyData) -> pv.PolyData:
    """
    wrap the BasePolyData object to a pyvista.PolyData object.

    Args:
        poly: BasePolyData

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
        return pv.PolyData(p, lines=new_m) if poly.is_linemesh else pv.PolyData(p, faces=new_m)

def _draw_mesh(plotter: pv.Plotter,
               mesh: CrossPoly,
               color: List[float],
               opacity: float = 1.0):
    """
    Draw a mesh to the plotter.

    Args:
        plotter: Plotter
        mesh: CrossPoly
        color: List[float]. The color of the mesh.
        opacity: float. The opacity of the mesh.
    """
    poly = _wrap_base_poly(mesh.get_polydata()) # PolyData object
    plotter.add_mesh(poly, color=color, opacity=opacity)
    
def _curve_list_to_polydata(meshes: List[Curve]) -> List[BasePolyData]:
    """Convert a list of Curve objects to a list of PolyData objects.

    Args:
        meshes: List of Curve objects

    Returns:
        List of PolyData objects
    """
    return [c.get_polydata() for c in meshes]

def draw_lens_3d(
    plotter: pv.Plotter,
    lens: GeoLens,
    save_dir: Optional[str] = None,
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
    # surf_color = np.array(surface_color)
    # sensor_color = np.array([0.5, 0.5, 0.5])
    surf_color = surface_color
    sensor_color = [0.5, 0.5, 0.5]

    
    # Create meshes
    surf_meshes, bridge_meshes, _, sensor_mesh = lens.create_mesh(
        mesh_rings, mesh_arms
    )

    # Draw meshes
    for surf in surf_meshes:
        if not isinstance(surf, Aperture):
            _draw_mesh(plotter, surf, color=surf_color, opacity=0.5)

    for bridge in bridge_meshes:
        _draw_mesh(plotter, bridge, color=surf_color, opacity=0.5)
    
    _draw_mesh(plotter, sensor_mesh, color=sensor_color, opacity=1.0)

    # Draw rays
    if draw_rays:
        rays_curve = geolens_ray_poly(lens, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms)
        
        rays_poly_list = [_curve_list_to_polydata(r) for r in rays_curve]
        rays_poly_fov = [merge(r) for r in rays_poly_list]
        rays_poly_fov = [_wrap_base_poly(r) for r in rays_poly_fov]
        for r in rays_poly_fov:
            plotter.add_mesh(r)

    # Save images
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plotter.show(screenshot=os.path.join(save_dir, "lens_layout3d.png"))
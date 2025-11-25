# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Utils for geometric lens systems.

Functions:
    - create_lens(): Create a lens design starting point with flat surfaces
    - create_surface(): Create a surface object based on the surface type
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deeplens.optics.geometric_surface import Aperture, Aspheric, AsphericNorm, Spheric, ThinLens, Plane
from deeplens.optics.materials import MATERIAL_data
from deeplens.basics import WAVE_RGB
from deeplens.geolens import GeoLens


# ====================================================================================
# Lens starting point generation
# ====================================================================================
def create_lens(
    foclen,
    fov,
    fnum,
    flange,
    enpd=None,
    thickness=None,
    surf_list=[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]],
    save_dir="./",
):
    """Create a lens design starting point with flat surfaces.

    Contributor: Rayengineer

    Args:
        foclen: Focal length in mm.
        fov: Diagonal field of view in degrees.
        fnum: Maximum f number.
        flange: Distance from last surface to sensor.
        thickness: Total thickness if specified.
        surf_list: List of surface types defining each lens element and aperture.
    """
    from deeplens.geolens import GeoLens

    # Compute lens parameters
    aper_r = foclen / fnum / 2
    imgh = round(2 * foclen * float(np.tan(np.deg2rad(fov / 2))), 2)
    if thickness is None:
        thickness = foclen + flange
    d_opt = thickness - flange

    # Materials
    mat_names = list(MATERIAL_data.keys())
    for mat in ["air", "vacuum", "occluder"]:
        if mat in mat_names:
            mat_names.remove(mat)

    # Create lens
    lens = GeoLens()
    surfaces = lens.surfaces

    d_total = 0.0
    for elem_type in surf_list:
        if elem_type == "Aperture":
            d_next = (torch.rand(1) + 0.5).item()
            surfaces.append(Aperture(r=aper_r, d=d_total))
            d_total += d_next

        elif isinstance(elem_type, list):
            if len(elem_type) == 1 and elem_type[0] == "Aperture":
                d_next = (torch.rand(1) + 0.5).item()
                surfaces.append(Aperture(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) == 1 and elem_type[0] == "ThinLens":
                d_next = (torch.rand(1) + 1.0).item()
                surfaces.append(ThinLens(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) in [2, 3]:
                for i, surface_type in enumerate(elem_type):
                    if i == len(elem_type) - 1:
                        mat = "air"
                        d_next = (torch.rand(1) + 0.5).item()
                    else:
                        mat = random.choice(mat_names)
                        d_next = (torch.rand(1) + 1.0).item()

                    surfaces.append(
                        create_surface(surface_type, d_total, aper_r, imgh, mat)
                    )
                    d_total += d_next
            else:
                raise Exception("Lens element type not supported yet.")
        else:
            raise Exception("Lens type format not correct.")

    # Normalize optical part total thickness
    d_opt_actual = d_total - d_next
    for s in surfaces:
        s.d = s.d / d_opt_actual * d_opt

    # Lens sensor (dummy sensor resolution)
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness).to(lens.device)
    lens.r_sensor = imgh / 2
    lens.set_sensor_res(sensor_res=(2000, 2000))
    
    # Lens calculation
    lens.enpd = enpd
    lens.float_enpd = True if enpd is None else False
    lens.float_foclen = False
    lens.float_rfov = False
    lens.post_computation()

    # Save lens
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}"
    lens.write_lens_json(os.path.join(save_dir, f"{filename}.json"))
    lens.analysis(os.path.join(save_dir, f"{filename}"))

    return lens

def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """Create a surface object based on the surface type."""
    if mat == "air":
        c = -float(np.random.rand()) * 0.001
    else:
        c = float(np.random.rand()) * 0.001
    r = max(imgh / 2, aper_r)

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)
    
    elif surface_type == "Aspheric":
        ai = np.random.randn(8).astype(np.float32) * 1e-24
        k = float(np.random.rand()) * 1e-6
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)

    elif surface_type == "Plane":
        return Plane(r=r, d=d_total, mat2=mat)
    
    else:
        raise Exception("Surface type not supported yet.")



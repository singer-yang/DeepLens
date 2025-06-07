"""Utils for GeoLens class."""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .optics.geometric_surface import Aperture, Aspheric, Spheric, ThinLens
from .optics.materials import MATERIAL_data
from .optics.basics import WAVE_RGB
from deeplens.geolens import GeoLens


# ====================================================================================
# ZEMAX file IO
# ====================================================================================
def read_zmx(filename="./test.zmx"):
    """Load the lens from .zmx file."""
    # Initialize a GeoLens
    from .geolens import GeoLens

    geolens = GeoLens()

    # Read .zmx file
    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(filename, "r", encoding="utf-16") as file:
            lines = file.readlines()

    # Iterate through the lines and extract SURF dict
    surfs_dict = {}
    current_surf = None
    for line in lines:
        if line.startswith("SURF"):
            current_surf = int(line.split()[1])
            surfs_dict[current_surf] = {}
        elif current_surf is not None and line.strip() != "":
            if len(line.strip().split(maxsplit=1)) == 1:
                continue
            else:
                key, value = line.strip().split(maxsplit=1)
                if key == "PARM":
                    new_key = "PARM" + value.split()[0]
                    new_value = value.split()[1]
                    surfs_dict[current_surf][new_key] = new_value
                else:
                    surfs_dict[current_surf][key] = value
        elif line.startswith("FLOA") or line.startswith("ENPD"):
            if line.startswith("FLOA"):
                geolens.float_enpd = True
                geolens.enpd = None
            else:
                geolens.float_enpd = False
                geolens.enpd = float(line.split()[1])

    geolens.float_foclen = False
    geolens.float_hfov = False
    # Read the extracted data from each SURF
    geolens.surfaces = []
    d = 0.0
    for surf_idx, surf_dict in surfs_dict.items():
        if surf_idx > 0 and surf_idx < current_surf:
            mat2 = (
                f"{surf_dict['GLAS'].split()[3]}/{surf_dict['GLAS'].split()[4]}"
                if "GLAS" in surf_dict
                else "air"
            )
            surf_r = float(surf_dict["DIAM"].split()[0]) if "DIAM" in surf_dict else 1.0
            surf_c = float(surf_dict["CURV"].split()[0]) if "CURV" in surf_dict else 0.0
            surf_d_next = (
                float(surf_dict["DISZ"].split()[0]) if "DISZ" in surf_dict else 0.0
            )

            if surf_dict["TYPE"] == "STANDARD":
                # Aperture
                if surf_c == 0.0 and mat2 == "air":
                    s = Aperture(r=surf_r, d=d)

                # Spherical surface
                else:
                    s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2)

            # Aspherical surface
            elif surf_dict["TYPE"] == "EVENASPH":
                raise NotImplementedError()
                s = Aspheric()

            else:
                print(f"Surface type {surf_dict['TYPE']} not implemented.")
                continue

            geolens.surfaces.append(s)
            d += surf_d_next

        elif surf_idx == current_surf:
            # Image sensor
            geolens.r_sensor = float(surf_dict["DIAM"].split()[0])

        else:
            pass

    geolens.d_sensor = torch.tensor(d)
    return geolens


def write_zmx(geolens, filename="./test.zmx"):
    """Write the lens into .zmx file."""
    lens_zmx_str = ""
    if geolens.float_enpd == True:
        enpd_str = 'FLOA'
    else:
        enpd_str = f'ENPD {geolens.enpd}'
    # Head string
    head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
{enpd_str}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707 * geolens.hfov * 57.3} {0.99 * geolens.hfov * 57.3}
WAVL 0.4861327 0.5875618 0.6562725
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
TYPE STANDARD
CURV 0.0
DISZ INFINITY
"""
    lens_zmx_str += head_str

    # Surface string
    for i, s in enumerate(geolens.surfaces):
        d_next = (
            geolens.surfaces[i + 1].d - geolens.surfaces[i].d
            if i < len(geolens.surfaces) - 1
            else geolens.d_sensor - geolens.surfaces[i].d
        )
        surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
        lens_zmx_str += surf_str

    # Sensor string
    sensor_str = f"""SURF {i + 2}
TYPE STANDARD
CURV 0.
DISZ 0.0
DIAM {geolens.r_sensor}
"""
    lens_zmx_str += sensor_str

    # Write lens zmx string into file
    with open(filename, "w") as f:
        f.writelines(lens_zmx_str)
        f.close()


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
    lens_type=[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]],
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
        lens_type: List of surface types defining each lens element and aperture.
    """
    from .geolens import GeoLens

    # Compute lens parameters
    aper_r = foclen / fnum / 2
    imgh = 2 * foclen * float(np.tan(np.deg2rad(fov / 2)))
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
    for elem_type in lens_type:
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

    # Lens calculation
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness).to(lens.device)
    lens.enpd = enpd
    lens.float_enpd = True if enpd is None else False
    lens.float_foclen = False
    lens.float_hfov = False
    lens.set_sensor(sensor_res=lens.sensor_res, r_sensor=imgh / 2)
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
        ai = np.random.randn(7).astype(np.float32) * 1e-30
        k = float(np.random.rand()) * 0.001
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)
    else:
        raise Exception("Surface type not supported yet.")



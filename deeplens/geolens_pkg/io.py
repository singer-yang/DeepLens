# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Lens file IO (ZEMAX)."""

import torch

from deeplens.optics.geometric_surface.aperture import Aperture
from deeplens.optics.geometric_surface.spheric import Spheric
from deeplens.optics.geometric_surface.aspheric import Aspheric

class GeoLensIO:
    def read_lens_zmx(self, filename="./test.zmx"):
        """Load the lens from .zmx file."""
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
                    self.float_enpd = True
                    self.enpd = None
                else:
                    self.float_enpd = False
                    self.enpd = float(line.split()[1])

        self.float_foclen = False
        self.float_hfov = False
        
        # Read the extracted data from each SURF
        self.surfaces = []
        d = 0.0
        for surf_idx, surf_dict in surfs_dict.items():
            if surf_idx > 0 and surf_idx < current_surf:
                # Lens surface parameters
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
                surf_conic = surf_dict.get("CONI", 0.0)
                surf_param2 = surf_dict.get("PARM2", 0.0)
                surf_param3 = surf_dict.get("PARM3", 0.0)
                surf_param4 = surf_dict.get("PARM4", 0.0)
                surf_param5 = surf_dict.get("PARM5", 0.0)
                surf_param6 = surf_dict.get("PARM6", 0.0)
                surf_param7 = surf_dict.get("PARM7", 0.0)
                surf_param8 = surf_dict.get("PARM8", 0.0)

                if surf_dict["TYPE"] == "STANDARD":
                    # Aperture
                    if surf_c == 0.0 and mat2 == "air":
                        s = Aperture(r=surf_r, d=d)

                    # Spherical surface
                    else:
                        s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2)

                # Aspherical surface
                elif surf_dict["TYPE"] == "EVENASPH":
                    s = Aspheric(c=surf_c, r=surf_r, d=d, ai=[surf_param2, surf_param3, surf_param4, surf_param5, surf_param6, surf_param7, surf_param8], k=surf_conic, mat2=mat2)

                else:
                    print(f"Surface type {surf_dict['TYPE']} not implemented.")
                    continue

                self.surfaces.append(s)
                d += surf_d_next

            elif surf_idx == current_surf:
                # Image sensor
                self.r_sensor = float(surf_dict["DIAM"].split()[0])

            else:
                pass
        
        self.d_sensor = torch.tensor(d)
        return self


    def write_lens_zmx(self, filename="./test.zmx"):
        """Write the lens into .zmx file."""
        lens_zmx_str = ""
        if self.float_enpd:
            enpd_str = 'FLOA'
        else:
            enpd_str = f'ENPD {self.enpd}'
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
    YFLN 0.0 {0.707 * self.hfov * 57.3} {0.99 * self.hfov * 57.3}
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
        for i, s in enumerate(self.surfaces):
            d_next = (
                self.surfaces[i + 1].d - self.surfaces[i].d
                if i < len(self.surfaces) - 1
                else self.d_sensor - self.surfaces[i].d
            )
            surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
            lens_zmx_str += surf_str

        # Sensor string
        sensor_str = f"""SURF {i + 2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_sensor}
    """
        lens_zmx_str += sensor_str

        # Write lens zmx string into file
        with open(filename, "w") as f:
            f.writelines(lens_zmx_str)
            f.close()

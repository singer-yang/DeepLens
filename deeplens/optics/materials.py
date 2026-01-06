# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Glass and plastic materials for optical lenses."""

import json
import os
import re

import torch

from deeplens.basics import DeepObj


# ===========================================
# Read AGF file
# ===========================================
def read_agf(file_path):
    """Read the AGF file and return the materials data."""
    encodings = ["utf-8", "utf-16"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
                break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Error! {file_path} not found.")

    nm_lines = [line for line in lines if re.match(r"^NM\b", line)]
    cd_lines = [line for line in lines if re.match(r"^CD\b", line)]

    materials = {}
    for i in range(len(nm_lines)):
        nm_parts = nm_lines[i].strip().split()
        cd_parts = cd_lines[i].strip().split()

        materials[nm_parts[1].lower()] = {
            "calculate_mode": float(nm_parts[2]),
            "nd": float(nm_parts[4]),
            "vd": float(nm_parts[5]),
            "a_coeff": float(cd_parts[1]),
            "b_coeff": float(cd_parts[2]),
            "c_coeff": float(cd_parts[3]),
            "d_coeff": float(cd_parts[4]),
            "e_coeff": float(cd_parts[5]),
            "f_coeff": float(cd_parts[6]),
        }
    return materials


CDGM_data = read_agf(os.path.dirname(__file__) + "/material/CDGM.AGF")
SCHOTT_data = read_agf(os.path.dirname(__file__) + "/material/SCHOTT.AGF")
MISC_data = read_agf(os.path.dirname(__file__) + "/material/MISC.AGF")
PLASTIC_data = read_agf(os.path.dirname(__file__) + "/material/PLASTIC2022.AGF")
MATERIAL_data = {**MISC_data, **PLASTIC_data, **CDGM_data, **SCHOTT_data}


# ===========================================
# Read custom materials from JSON file
# ===========================================
def read_custom_mat(file_path):
    """Read the JSON file and return the materials data."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Materials data file not found at {file_path}")
        return {}


CUSTOM_data = read_custom_mat(
    os.path.dirname(__file__) + "/material/materials_data.json"
)


# ===========================================
# Material class
# ===========================================
class Material(DeepObj):
    def __init__(self, name=None, device="cpu"):
        self.name = "vacuum" if name is None else name.lower()
        self.load_dispersion()
        self.device = device

    def get_name(self):
        if self.dispersion == "optimizable":
            return f"{self.n.item():.4f}/{self.V.item():.2f}"
        else:
            return self.name

    # -------------------------------------------
    # Load dispersion equation
    # -------------------------------------------
    def load_dispersion(self):
        """Load material dispersion equation."""
        # Air, vacuum, occluder are special cases
        if self.name == "air" or self.name == "vacuum" or self.name == "occluder":
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
            self.n, self.V = 1.0, 1e38

        # Material found in AGF file
        elif self.name.lower() in MATERIAL_data:
            self.set_material_param_agf(MATERIAL_data, self.name.lower())

        # Material is given by a (n, V) string, e.g. "1.5168/64.17"
        elif "/" in self.name:
            self.dispersion = "cauchy"
            self.n = float(self.name.split("/")[0])
            self.V = float(self.name.split("/")[1])
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        # Material found in custom JSON file
        elif self.name in CUSTOM_data["INTERP_TABLE"]:
            self.dispersion = "interp"
            self.ref_wvlns = CUSTOM_data["INTERP_TABLE"]["wvlns"]
            self.ref_n = CUSTOM_data["INTERP_TABLE"][self.name]
            self.n = sum(self.ref_n) / len(self.ref_n)

        elif self.name in CUSTOM_data["SELLMEIER_TABLE"]:
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = CUSTOM_data[
                "SELLMEIER_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SELLMEIER_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["SCHOTT_TABLE"]:
            self.dispersion = "schott"
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = CUSTOM_data[
                "SCHOTT_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SCHOTT_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["MATERIAL_TABLE"]:
            self.dispersion = "cauchy"
            self.n, self.V = CUSTOM_data["MATERIAL_TABLE"][self.name]
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        else:
            raise NotImplementedError(f"Material {self.name} not implemented.")

    def set_material_param_agf(self, material_data, material_name):
        """Set the material parameters and dispersion equation from AGF file."""
        if material_name in material_data:
            material = material_data[material_name]

            if material["calculate_mode"] == 1:
                self.dispersion = "schott"
                self.a0 = material["a_coeff"]
                self.a1 = material["b_coeff"]
                self.a2 = material["c_coeff"]
                self.a3 = material["d_coeff"]
                self.a4 = material["e_coeff"]
                self.a5 = material["f_coeff"]
            elif material["calculate_mode"] == 2:
                self.dispersion = "sellmeier"
                self.k1 = material["a_coeff"]
                self.l1 = material["b_coeff"]
                self.k2 = material["c_coeff"]
                self.l2 = material["d_coeff"]
                self.k3 = material["e_coeff"]
                self.l3 = material["f_coeff"]
            else:
                raise NotImplementedError(
                    f"Error: {material_name} calculate_mode {material['calculate_mode']}"
                )

            self.n = material["nd"]
            self.V = material["vd"]
        else:
            print(f"error: not {material_name}")

    def set_sellmeier_param(self, params=None):
        """Manually set sellmeier parameters k1, l1, k2, l2, k3, l3.

        This function is used when we want to manually set the sellmeier parameters for a custom material.
        """
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params

    # -------------------------------------------
    # Calculate refractive index
    # -------------------------------------------
    def refractive_index(self, wvln):
        """Compute the refractive index at given wvln."""
        if isinstance(wvln, float):
            wvln = torch.tensor(wvln).to(self.device)
            return self.ior(wvln).item()

        return self.ior(wvln)

    def ior(self, wvln):
        """Compute the refractive index at given wvln."""
        assert wvln.min() > 0.1 and wvln.max() < 10, "Wavelength should be in [um]."

        if self.dispersion == "sellmeier":
            # Sellmeier equation: https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = (
                1
                + self.k1 * wvln**2 / (wvln**2 - self.l1)
                + self.k2 * wvln**2 / (wvln**2 - self.l2)
                + self.k3 * wvln**2 / (wvln**2 - self.l3)
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "schott":
            # Schott equation: https://johnloomis.org/eop501/notes/matlab/sect1/schott.html
            ws = wvln**2
            n2 = (
                self.a0
                + self.a1 * ws
                + (self.a2 + (self.a3 + (self.a4 + self.a5 / ws) / ws) / ws) / ws
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "cauchy":
            # Cauchy equation: https://en.wikipedia.org/wiki/Cauchy%27s_equation
            n = self.A + self.B / (wvln * 1e3) ** 2

        elif self.dispersion == "interp":
            # Convert reference wavelengths and refractive indices to tensors
            ref_wvlns = torch.tensor(self.ref_wvlns, device=wvln.device)
            ref_n = torch.tensor(self.ref_n, device=wvln.device)

            # Find the lower and upper bracketing wavelengths
            i = torch.searchsorted(ref_wvlns, wvln, side="right")
            num_ref_wvlns = len(ref_wvlns)
            idx_low = torch.clamp(i - 1, 0, num_ref_wvlns - 1)
            idx_high = torch.clamp(i, 0, num_ref_wvlns - 1)

            wvln_ref_low = ref_wvlns[idx_low]
            wvln_ref_high = ref_wvlns[idx_high]
            n_ref_low = ref_n[idx_low]
            n_ref_high = ref_n[idx_high]

            # Interpolate n
            weight_high = (wvln - wvln_ref_low) / (wvln_ref_high - wvln_ref_low)
            weight_low = 1.0 - weight_high
            n = n_ref_low * weight_low + n_ref_high * weight_high

        elif self.dispersion == "optimizable":
            # Cauchy's equation, calculate (A, B) on the fly
            B = (self.n - 1) / self.V / (1 / 0.486**2 - 1 / 0.656**2)
            A = self.n - B * 1 / 0.587**2
            n = A + B / wvln**2

        else:
            raise NotImplementedError(f"Error: {self.dispersion} not implemented.")

        return n

    @staticmethod
    def nV_to_AB(n, V):
        """Convert (n ,V) paramters to (A, B) parameters to find the material."""

        def ivs(a):
            return 1.0 / a**2

        lambdas = [656.3, 587.6, 486.1]
        B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))
        A = n - B * ivs(lambdas[1])
        return A, B

    # -------------------------------------------
    # Optimize and match material
    # -------------------------------------------
    def match_material(self, mat_table=None):
        """Find the closest material in the CDGM common glasses database."""
        if not self.name == "air":
            # Material match table
            if mat_table is None:
                print("No material table provided. Using CDGM common glasses as default.")
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "CDGM":
                # CDGM common glasses
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "PLASTIC":
                mat_table = CUSTOM_data["PLASTIC_TABLE"]
            else:
                raise NotImplementedError(f"Material table {mat_table} not implemented.")

            # Find the closest material
            n_range = 0.4 # refractive index range usually [1.5, 1.9]
            V_range = 40.0 # Abbe number range usually [30, 70]
            dist_min = 1e6
            for name in mat_table:
                n, V = mat_table[name]
                error_n = abs(n - self.n) / n_range
                error_V = abs(V - self.V) / V_range
                dist = error_n + error_V
                if dist < dist_min:
                    self.name = name
                    dist_min = dist

            # Load the new material parameters
            self.load_dispersion()

    def get_optimizer_params(self, lrs=[1e-4, 1e-2]):
        """Optimize the material parameters (n, V). 
        
        Optimizing refractive index is more important than optimizing Abbe number.
        
        Args:
            lrs (list): learning rates for n and V. Defaults to [1e-4, 1e-4].
        """
        if isinstance(self.n, float):
            self.n = torch.tensor(self.n).to(self.device)
            self.V = torch.tensor(self.V).to(self.device)

        self.n.requires_grad = True
        self.V.requires_grad = True
        self.dispersion = "optimizable"

        params = [
            {"params": [self.n], "lr": lrs[0]},
            {"params": [self.V], "lr": lrs[1]},
        ]
        return params

"""Glass and plastic materials."""

import json
import os
import torch
import re
from .basics import DeepObj

# Load materials data from AGF file
def read_agf(file_path):
    encodings = ['utf-8', 'utf-16']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = f.readlines()
                break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("error!")

    nm_lines = [line for line in lines if re.match(r'^NM\b', line)]
    cd_lines = [line for line in lines if re.match(r'^CD\b', line)]

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
            "f_coeff": float(cd_parts[6])
        }
    return materials

CDGM_data = read_agf(os.path.dirname(__file__) + "/material/CDGM.AGF")
SCHOTT_data = read_agf(os.path.dirname(__file__) + "/material/SCHOTT.AGF")
MISC_data = read_agf(os.path.dirname(__file__) + "/material/MISC.AGF")
PLASTIC_data = read_agf(os.path.dirname(__file__) + "/material/PLASTIC2022.AGF")
MATERIAL_data = {**MISC_data, **PLASTIC_data, **CDGM_data, **SCHOTT_data}

# Load materials data from JSON file
MATERIALS_DATA_PATH = os.path.join(os.path.dirname(__file__), "materials_data.json")
try:
    with open(MATERIALS_DATA_PATH, "r") as f:
        MATERIALS_DATA = json.load(f)

    # Extract tables from the loaded data
    MATERIAL_TABLE = MATERIALS_DATA.get("MATERIAL_TABLE", {})
    SELLMEIER_TABLE = MATERIALS_DATA.get("SELLMEIER_TABLE", {})
    SCHOTT_TABLE = MATERIALS_DATA.get("SCHOTT_TABLE", {})
    INTERP_TABLE = MATERIALS_DATA.get("INTERP_TABLE", {})
    GLASS_NAME = MATERIALS_DATA.get("GLASS_NAME", {})
    CDGM_GLASS = MATERIALS_DATA.get("CDGM_GLASS", {})

except FileNotFoundError:
    print(f"Warning: Materials data file not found at {MATERIALS_DATA_PATH}")


class Material(DeepObj):
    def __init__(self, name=None, device="cpu"):
        self.name = "vacuum" if name is None else name.lower()
        self.load_dispersion()
        self.device = device

    def set_material_parameter(self, material_data, material_name):

        if material_name in material_data:
            material = material_data[material_name]

            if material['calculate_mode'] == 1:
                self.dispersion = "schott"
                self.a0 = material['a_coeff']
                self.a1 = material['b_coeff']
                self.a2 = material['c_coeff']
                self.a3 = material['d_coeff']
                self.a4 = material['e_coeff']
                self.a5 = material['f_coeff']
            elif material['calculate_mode'] == 2:
                self.dispersion = "sellmeier"
                self.k1 = material['a_coeff']
                self.l1 = material['b_coeff']
                self.k2 = material['c_coeff']
                self.l2 = material['d_coeff']
                self.k3 = material['e_coeff']
                self.l3 = material['f_coeff']
            else:
                raise NotImplementedError(f"error: {material_name} calculate_mode {material['calculate_mode']}")

            self.n = material['nd']
            self.V = material['vd']
        else:
            print(f"error: not {material_name}")

    def get_name(self):
        if self.dispersion == "optimizable":
            return f"{self.n.item():.4f}/{self.V.item():.2f}"
        else:
            return self.name

    def load_dispersion(self):
        """Load material dispersion equation."""
        if self.name == 'air' or self.name == 'vacuum' or self.name == 'occluder':
            self.dispersion = "sellmeier"
            self.k1,self.l1, self.k2, self.l2, self.k3, self.l3 = 0,0,0,0,0,0
            self.n, self.V = 1, 1e38

        elif self.name in MATERIAL_data:
            self.set_material_parameter(MATERIAL_data, self.name)

        elif self.name in INTERP_TABLE:
            self.dispersion = "interp"
            self.ref_wvlns = INTERP_TABLE["wvlns"]
            self.ref_n = INTERP_TABLE[self.name]

        else:
            self.dispersion = "cauchy"
            self.n, self.V = (
                float(self.name.split("/")[0]),
                float(self.name.split("/")[1]),
            )
            self.A, self.B = self.nV_to_AB(self.n, self.V)

    def refractive_index(self, wvln):
        """Compute the refractive index at given wvln."""
        return self.ior(wvln)

    def ior(self, wvln):
        """Compute the refractive index at given wvln."""
        assert wvln > 0.1 and wvln < 1, "Wavelength should be in [um]."

        if self.dispersion == "sellmeier":
            # Sellmeier equation
            # https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = (
                1
                + self.k1 * wvln**2 / (wvln**2 - self.l1)
                + self.k2 * wvln**2 / (wvln**2 - self.l2)
                + self.k3 * wvln**2 / (wvln**2 - self.l3)
            )
            n = torch.sqrt(torch.tensor(n2)).item()

        elif self.dispersion == "schott":
            # High precision computation (by MATLAB), writing dispersion function seperately will introduce errors
            ws = wvln**2
            n2 = (
                self.a0
                + self.a1 * ws
                + (self.a2 + (self.a3 + (self.a4 + self.a5 / ws) / ws) / ws) / ws
            )
            n = torch.sqrt(torch.tensor(n2)).item()

        elif self.dispersion == "cauchy":
            # Cauchy equation
            # https://en.wikipedia.org/wiki/Cauchy%27s_equation
            n = self.A + self.B / (wvln * 1e3) ** 2

        elif self.dispersion == "interp":
            ref_wvlns = self.ref_wvlns
            ref_n = self.ref_n

            # Find the nearest two wvlns
            delta_wvln = [abs(wv - wvln) for wv in ref_wvlns]
            idx1 = delta_wvln.index(min(delta_wvln))
            delta_wvln[idx1] = float("inf")
            idx2 = delta_wvln.index(min(delta_wvln))

            # Interpolate n
            n = ref_n[idx1] + (ref_n[idx2] - ref_n[idx1]) / (
                ref_wvlns[idx2] - ref_wvlns[idx1]
            ) * (wvln - ref_wvlns[idx1])

        elif self.dispersion == "optimizable":
            # Cauchy's equation, calculate (A, B) on the fly
            B = (self.n - 1) / self.V / (1 / 0.486**2 - 1 / 0.656**2)
            A = self.n - B * 1 / 0.589**2

            n = A + B / wvln**2

        else:
            raise NotImplementedError

        return n

    def load_sellmeier_param(self, params=None):
        """Manually set sellmeier parameters k1, l1, k2, l2, k3, l3.

        This function is called when we want to use a custom material.
        """
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params

    @staticmethod
    def nV_to_AB(n, V):
        """Convert (n ,V) paramters to (A, B) parameters to find the material."""

        def ivs(a):
            return 1.0 / a**2

        lambdas = [656.3, 589.3, 486.1]
        B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))
        A = n - B * ivs(lambdas[1])
        return A, B

    def match_material(self, mat_table=None):
        """Find the closest material in the database."""
        if mat_table is None:
            mat_table = MATERIAL_TABLE
        elif mat_table == "CDGM":
            mat_table = CDGM_GLASS
        else:
            raise NotImplementedError

        weight_n = 2
        dist_min = 1e6
        for name in mat_table:
            n, V = mat_table[name]
            dist = weight_n * abs(n - self.n) / self.n + abs(V - self.V) / self.V
            if dist < dist_min:
                self.name = name
                dist_min = dist

        self.load_dispersion()

    def get_optimizer_params(self, lr=[1e-5, 1e-3]):
        """Optimize the material parameters (n, V)."""
        if isinstance(self.n, float):
            self.n = torch.tensor(self.n).to(self.device)
            self.V = torch.tensor(self.V).to(self.device)

        self.n.requires_grad = True
        self.V.requires_grad = True
        self.dispersion = "optimizable"

        params = [{"params": [self.n], "lr": lr[0]}, {"params": [self.V], "lr": lr[1]}]
        return params

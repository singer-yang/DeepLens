"""Basic variables and classes for DeepLens."""

import copy

import numpy as np
import torch
import torch.nn as nn

def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name}")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print("Using CPU")
    return device

# ===========================================
# Variables
# ===========================================
DEPTH = -20000.0 # approximate infinity

SPP_PSF = 10000 # spp (sample per pixel) for psf calculation
SPP_COHERENT = 10000000 # spp for coherent optics calculation
SPP_CALC = 1024 # spp for some computation which doesnot need to be very accurate, e.g., refocusing
SPP_RENDER = 32 # spp for rendering

PSF_KS = 101 # kernel size for psf calculation, better to be odd number
GEO_GRID = 21  # grid number for PSF map

DELTA = 1e-6
EPSILON = 1e-9  # replace 0 with EPSILON in some cases

DEFAULT_WAVE = 0.589 # [um] default wavelength
WAVE_RGB = [0.656, 0.589, 0.486] # [um] R, G, B wavelength

WAVE_RED = [0.620, 0.660, 0.700] # [um] narrow band red spectrum
WAVE_GREEN = [0.500, 0.530, 0.560] # [um] narrow band green spectrum
WAVE_BLUE = [0.450, 0.470, 0.490] # [um] narrow band blue spectrum

WAVE_BOARD_BAND = [
    0.400,
    0.410,
    0.420,
    0.430,
    0.440,
    0.450,
    0.460,
    0.470,
    0.480,
    0.490,
    0.500,
    0.510,
    0.520,
    0.530,
    0.540,
    0.550,
    0.560,
    0.570,
    0.580,
    0.590,
    0.600,
    0.610,
    0.620,
    0.630,
    0.640,
    0.650,
    0.660,
    0.670,
    0.680,
    0.690,
    0.700,
]

RED_RESPONSE = [
    0.00,
    0.00,
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.05,
    0.05,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.15,
    0.20,
    0.40,
    0.50,
    0.60,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.15,
    0.10,
    0.05,
    0.03,
    0.00,
]

GREEN_RESPONSE = [
    0.00,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.00,
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.30,
    0.20,
    0.10,
    0.05,
    0.04,
    0.03,
    0.02,
    0.01,
    0.00,
    0.00,
    0.00,
    0.00,
]

BLUE_RESPONSE = [
    0.00,
    0.30,
    0.60,
    0.70,
    0.80,
    0.90,
    0.80,
    0.60,
    0.40,
    0.30,
    0.20,
    0.10,
    0.05,
    0.03,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
]


WAVE_SPEC = [
    0.400,
    0.420,
    0.440,
    0.460,
    0.480,
    0.500,
    0.520,
    0.540,
    0.560,
    0.580,
    0.600,
    0.620,
    0.640,
    0.660,
    0.680,
    0.700,
]
FULL_SPECTRUM = np.arange(0.400, 0.701, 0.02)
HYPER_SPEC_RANGE = [0.42, 0.66]  # [um]. reference 400nm to 700nm, 20nm step size
HYPER_SPEC_BAND = 49  # 5nm/step, according to "Shift-variant color-coded diffractive spectral imaging system"


def wave_rgb():
    """Randomly select one wave from R, G, B spectrum and return the wvln list (length 3)"""
    wave_r = np.random.choice([0.620, 0.660, 0.700])
    wave_g = np.random.choice([0.500, 0.530, 0.560])
    wave_b = np.random.choice([0.450, 0.470, 0.490])
    return [wave_r, wave_g, wave_b]


# ===========================================
# Classes
# ===========================================
class DeepObj:
    def __str__(self):
        """Called when using print() and str()"""
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if val.__class__.__name__ in ["list", "tuple"]:
                for i, v in enumerate(val):
                    lines += "{}[{}]: {}".format(key, i, v).split("\n")
            elif val.__class__.__name__ in ["dict", "OrderedDict", "set"]:
                pass
            else:
                lines += "{}: {}".format(key, val).split("\n")

        return "\n    ".join(lines)

    def __call__(self, inp):
        """Call the forward function."""
        return self.forward(inp)

    def clone(self):
        """Clone a DeepObj object."""
        return copy.deepcopy(self)

    def to(self, device):
        """Move all variables to target device.

        Args:
            device (cpu or cuda, optional): target device. Defaults to torch.device('cpu').
        """
        self.device = device

        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec(f"self.{key} = self.{key}.to(device)")
            elif isinstance(val, nn.Module):
                exec(f"self.{key}.to(device)")
            elif issubclass(type(val), DeepObj):
                exec(f"self.{key}.to(device)")
            elif val.__class__.__name__ in ("list", "tuple"):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(f"self.{key}[{i}] = self.{key}[{i}].to(device)")
                    elif issubclass(type(v), DeepObj):
                        exec(f"self.{key}[{i}].to(device)")
        return self

    def double(self):
        """Convert all float32 tensors to float64.

        Remember to upgrade pytorch to the latest version to use double-precision optimizer.

        torch.set_default_dtype(torch.float64)
        """
        assert (
            torch.get_default_dtype() == torch.float64
        ), "Default dtype should be float64."

        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec(f"self.{key} = self.{key}.double()")
            elif isinstance(val, nn.Module):
                exec(f"self.{key}.double()")
            elif issubclass(type(val), DeepObj):
                exec(f"self.{key}.double()")
            elif issubclass(type(val), list):
                # Now only support tensor list or DeepObj list
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(f"self.{key}[{i}] = self.{key}[{i}].double()")
                    elif issubclass(type(v), DeepObj):
                        exec(f"self.{key}[{i}].double()")

        return self

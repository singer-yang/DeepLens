"""Basic variables and classes for DeepLens."""

import copy

import numpy as np
import torch
import torch.nn as nn

def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name} for DeepLens")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        print("Using MPS (Apple Silicon) for DeepLens")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print("Using CPU for DeepLens")
    return device

# ===========================================
# Variables
# ===========================================
DEPTH = -20000.0 # approximate infinity

SPP_PSF = 2 << 13 # 16384, spp (sample per pixel) for psf calculation
SPP_COHERENT = 2 << 23 # 1.67*10^7, spp for coherent optics calculation
SPP_CALC = 1024 # spp for some computation which doesnot need to be very accurate, e.g., refocusing
SPP_RENDER = 32 # spp for rendering
SPP_PARAXIAL = 32 # spp for paraxial

PSF_KS = 64 # kernel size for psf calculation
GEO_GRID = 21  # grid number for PSF map

DELTA = 1e-6
DELTA_PARAXIAL = 0.01
EPSILON = 1e-12  # replace 0 with EPSILON in some cases

DEFAULT_WAVE = 0.587 # [um] default wavelength
WAVE_RGB = [0.656, 0.587, 0.486] # [um] R, G, B wavelength

WAVE_RED = [0.620, 0.660, 0.700] # [um] narrow band red spectrum
WAVE_GREEN = [0.500, 0.530, 0.560] # [um] narrow band green spectrum
WAVE_BLUE = [0.450, 0.470, 0.490] # [um] narrow band blue spectrum

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
    def __init__(self, dtype=None):
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

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


    def astype(self, dtype):
        """Convert all tensors to the given dtype.

        Args:
            dtype (torch.dtype): Data type.
        """
        if dtype is None:
            return self
        
        dtype_ls = [torch.float16, torch.float32, torch.float64]
        assert dtype in dtype_ls, f"Data type {dtype} is not supported."
        
        if torch.get_default_dtype() != dtype:
            torch.set_default_dtype(dtype)
            print(f"Set {dtype} as default torch dtype.")
        
        self.dtype = dtype
        for key, val in vars(self).items():
            if torch.is_tensor(val) and val.dtype in dtype_ls:
                exec(f"self.{key} = self.{key}.to(dtype)")
            elif issubclass(type(val), DeepObj):
                exec(f"self.{key}.astype(dtype)")
            elif issubclass(type(val), list):
                for i, v in enumerate(val):
                    if torch.is_tensor(v) and v.dtype in dtype_ls:
                        exec(f"self.{key}[{i}] = self.{key}[{i}].to(dtype)")
                    elif issubclass(type(v), DeepObj):
                        exec(f"self.{key}[{i}].astype(dtype)")
        return self
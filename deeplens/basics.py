"""
This file contains basic classes and variables for DeepLens.

(1), Viriables, material tables
(2), Base class, ray class, material class
"""
import torch
import math
import copy
import numpy as np
import torch.nn as nn
from enum import Enum
import torch.nn.functional as nnF


# ===========================================
# Variables
# ===========================================
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

DEFAULT_WAVE = 0.589
WAVE_RGB = [0.656, 0.589, 0.486]
WAVE_SPEC = [0.400, 0.420, 0.440, 0.460, 0.480, 0.500, 0.520, 0.540, 0.560, 0.580, 0.600, 0.620, 0.640, 0.660, 0.680, 0.700]
FULL_SPECTRUM = np.arange(0.400, 0.701, 0.02)
HYPER_SPEC_RANGE = [0.42, 0.66] # [um]. reference 400nm to 700nm, 20nm step size
HYPER_SPEC_BAND = 49    # 5nm/step, according to "Shift-variant color-coded diffractive spectral imaging system"

DEPTH = - 20000.0
GEO_SPP = 2048   # spp for geometric optics calculation (psf)
GEO_GRID = 21   # grid number for geometric optics calculation (PSF map)
COHERENT_SPP = 10000000 # spp for coherent optics calculation

MINT = 1e-5
MAXT = 1e5
DELTA = 1e-6
EPSILON = 1e-9  # replace 0 with EPSILON in some cases
NEWTON_STEP_BOUND = 1   # Maximum step length in one Newton iteration


# We can find off-the-shelf glasses here:
# https://www.schott.com/en-dk/interactive-abbe-diagram
# https://refractiveindex.info/ 
MATERIAL_TABLE = { 
    # [nD, Abbe number]
    "vacuum":       [1.,       math.inf],
    "air":          [1.,       math.inf],
    "occluder":     [1.,       math.inf],
    "f2":           [1.620,    36.37],
    "f5":           [1.6034,   38.03],
    "bk1":          [1.5101,   63.47],
    "bk7":          [1.5168,   64.17],
    
    # https://shop.schott.com/advanced_optics/
    "bk10":         [1.49780,  66.954],
    "kzfs1":        [1.6131,   44.339],
    "laf20":        [1.6825,   48.201],
    "lafn7":        [1.7495,   34.951],
    "n-baf10":      [1.67003,  47.11],
    "n-bk7":        [1.51680,  64.17],
    "n-lak34":      [1.75500,  52.30],
    "n-pk51":       [1.53100,  56.00],
    "n-pk52":       [1.49700,  81.63],
    "n-balf4":      [1.57960,  53.86],
    "n-ssk2":       [1.62229,  53.27],
    "n-sf57":       [1.84666,  23.78],
    "n-sf10":       [1.72828,  28.53],
    "sf5":          [1.67270,  32.21],
    "sf11":         [1.87450,  25.68],
    "n-bak4":       [1.56883,  55.98],

    # plastic for cellphone
    # from paper: Analysis of the dispersion of optical plastic materials
    "coc":          [1.5337,   56.22],
    "pmma":         [1.4918,   57.44],
    "ps":           [1.5904,   30.87],
    "pc":           [1.5855,   29.91],
    "okp4ht":       [1.6328,   23.34],
    "okp4":         [1.6328,   23.34],
    
    "apl5014cl":    [1.5445,   55.987],
    "d-k59":        [1.5176,   63.500],

    # SUMITA.AGF
    "sk1":          [1.61020,  56.504],
    "sk16":         [1.62040,  60.306],
    "sk1":          [1.61030,  56.712],
    "sk16":         [1.62040,  60.324],
    "ssk4":         [1.61770,  55.116],

    # https://www.pgo-online.com/intl/B270.html
    "b270":         [1.52290,  58.50],
    
    # https://refractiveindex.info, nd at 589.3 [nm]
    "s-nph1":       [1.8078,   22.76], 
    "d-k59":        [1.5175,   63.50],
    "hk51":         [1.5501,   58.64],
    "d-zk3":        [1.5891,   61.15],
    "h-zf2":        [1.6727,   32.17],
    "h-lak51":      [1.6968,   55.53],
    
    "flint":        [1.6200,   36.37],
    "pmma":         [1.491756, 58.00],
    "polycarb":     [1.58547,  29.91],
    "polystyr":     [1.59048,  30.87]
}


# Sellmeier equation parameters. Reference: https://en.wikipedia.org/wiki/Sellmeier_equation
SELLMEIER_TABLE = {
    "vacuum":       [0., 0., 0., 0., 0., 0.],
    "air":          [0., 0., 0., 0., 0., 0.],
    "occluder":     [0., 0., 0., 0., 0., 0.],
    "f2":           [1.3453, 9.9774e-3, 2.0907e-1, 4.7045e-2, 9.3736e-1, 1.1188e2],
    "f5":           [1.3104, 9.5863e-3, 1.9603e-1, 4.5762e-2, 9.6612e-1, 1.1501e2],
    "bk1":          [1.0425, 6.1656e-3, 2.0838e-1, 2.1215e-2, 9.8014e-1, 1.0906e2],
    "bk7":          [1.0396, 6.0006e-3, 2.3179e-1, 2.0017e-2, 1.0104,    1.0356e2],
    "sf11":         [1.7385, 1.3607e-2, 3.1117e-1, 6.1596e-2, 1.1749,    1.2192e2],
    
    # https://shop.schott.com/advanced_optics/
    "kzfs1":        [1.3661, 8.7316e-3, 1.8204e-1, 3.8983e-2, 8.6431e-1, 6.2939e1],
    "laf20":        [1.6510, 9.7050e-3, 1.1847e-1, 4.2892e-2, 1.1154, 1.1405e2],
    "lafn7":        [1.6684, 1.0316e-2, 2.9851e-1, 4.6922e-2, 1.0774, 8.2508e1],
    "n-bk7":        [1.0396, 6.0006e-3, 2.3179e-1, 2.0017e-2, 1.0104, 1.0356e2],
    "n-lak34":      [1.2666, 5.8928e-3, 6.6592e-1, 1.9751e-2, 1.1247, 78.889],
    "n-pk51":       [1.1516, 5.8556e-3, 1.5323e-1, 1.9407e-2, 7.8562e-1, 140.537],
    "n-pk52":       [1.0081, 5.0197e-3, 2.0943e-1, 1.6248e-2, 7.8169e-1, 1.5239e2],
    "n-balf4":      [1.3100, 7.9659e-3, 1.4208e-1, 3.3067e-2, 9.6493e-1, 1.0919e2],

    # SUMITA.AGF
    "sk16":         [1.3431, 7.0468e-3, 2.4114e-1, 2.2900e-2, 9.9432e-1, 9.2751e1],

    # https://www.pgo-online.com/intl/B270.html
    
    # https://refractiveindex.info, nd at 589.3 [nm]
    "d-k59":        [1.1209, 6.5791e-3, 1.5269e-1, 2.3572e-2, 1.0750000, 1.0631e2],
    'hk51':         [0.9602, 116.24248, 1.1836896, 0.0118030, 0.1023382, 0.018958],
    "d-zk3":        [1.3394, 0.0076061, 0.1486902, 0.0238444, 1.0095403, 89.04198],
    "h-zf2":        [0.1676, 0.0605178, 1.5433507, 0.0118524, 1.1731312, 113.6711],
    "h-lak51":      [1.1875, 0.0158001, 0.6393986, 5.6713e-5, 1.2654535, 91.09705],
}


SCHOTT_TABLE = {
    "coc":          [2.28449,  1.02952e-2, 3.73494e-2, -9.28410e-3, 1.73290e-3, -1.15203e-4],
    "pmma":         [2.18646, -2.44753e-4, 1.41558e-2, -4.43298e-4, 7.76643e-5, -2.99364e-6],
    "ps":           [2.44598,  2.21429e-5, 2.72989e-2,  3.01211e-4, 8.88934e-5, -1.75708e-6],
    'polystyr':     [2.44598,  2.21429e-5, 2.72989e-2,  3.01211e-4, 8.88934e-5, -1.75708e-6],
    "pc":           [2.42839, -3.86117e-5, 2.87574e-2, -1.97897e-4, 1.48359e-4,  1.38652e-6],
    'polycarb':     [2.42839, -3.86117e-5, 2.87574e-2, -1.97897e-4, 1.48359e-4,  1.38652e-6],
    "okp4ht":       [2.55219,  6.51282e-5, 3.57452e-2,  8.49831e-4, 8.47777e-5,  1.58990e-5],
    "okp4":         [2.49230, -1.46713e-3, 3.04056e-2, -2.31960e-4, 3.62928e-4, -1.89103e-5],
}


# Refractive indices from 0.4um to 0.7um for interpolation, 0.01um step size
INTERP_TABLE = {
    "fused_silica": [1.4701, 1.4692, 1.4683, 1.4674, 1.4665, 1.4656, 1.4649, 1.4642, 1.4636, 1.4629, 1.4623, 1.4619, 1.4614, 1.4610, 1.4605, 1.4601, 1.4597, 1.4593, 1.4589, 1.4585, 1.4580, 1.4577, 1.4574, 1.4571, 1.4568, 1.4565, 1.4563, 1.4560, 1.4558, 1.4555, 1.4553],
}


GLASS_NAME = {
    "coc":          'COC',
    "pmma":         'PMMA',
    "ps":           'POLYSTYR',
    'polystyr':     'POLYSTYR',
    "pc":           'POLYCARB',
    "polycarb":     'POLYCARB',
    "okp4":         'OKP4',
    "okp4ht":       'OKP4HT'
}



# ===========================================
# Classes
# ===========================================
class DeepObj(nn.Module):  
    def __str__(self):
        """ Called when using print() and str()
        """
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if val.__class__.__name__ in ['list', 'tuple']:
                for i, v in enumerate(val):
                    lines += '{}[{}]: {}'.format(key, i, v).split('\n')
            elif val.__class__.__name__ in ['dict', 'OrderedDict', 'set']:
                pass
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        
        return '\n    '.join(lines)
    
    def __call__(self, inp):
        """ Call the forward function.
        """
        return self.forward(inp)
    
    def clone(self):
        """ Clone a DeepObj object.
        """
        return copy.deepcopy(self)
    
    def to(self, device=DEVICE):
        """ Move all variables to target device.

        Args:
            device (cpu or cuda, optional): target device. Defaults to torch.device('cpu').
        """
        self.device = device
        
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec(f'self.{key} = self.{key}.to(device)')
            elif isinstance(val, nn.Module):
                exec(f'self.{key}.to(device)')
            elif issubclass(type(val), DeepObj):
                exec(f'self.{key}.to(device)')
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(f'self.{key}[{i}] = self.{key}[{i}].to(device)')
                    elif issubclass(type(v), DeepObj):
                        exec(f'self.{key}[{i}].to(device)')
        return self
    
    def double(self):
        """ Convert all float32 tensors to float64.

            Remember to upgrade pytorch to the latest version to use double-precision optimizer.

            torch.set_default_dtype(torch.float64)
        """
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec(f'self.{key} = self.{key}.double()')
            elif isinstance(val, nn.Module):
                exec(f'self.{key}.double()')
            elif issubclass(type(val), DeepObj):
                exec(f'self.{key}.double()')
            elif issubclass(type(val), list):
                # Now only support tensor list or DeepObj list
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(f'self.{key}[{i}] = self.{key}[{i}].double()')
                    elif issubclass(type(v), DeepObj):
                        exec(f'self.{key}[{i}].double()')

        return self
    

class Ray(DeepObj):
    def __init__(self, o, d, wvln=DEFAULT_WAVE, coherent=False, device=DEVICE):
        """ Ray class. Optical rays with the same wvln.

        Args:
            o (Tensor): ray position. shape [..., 3]
            d (Tensor): normalized ray direction. shape [..., 3]
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            ra (Tensor, optional): Validity. Defaults to None.
            en (Tensor, optional): Spherical wave energy decay. Defaults to None.
            obliq (Tensor, optional): Obliquity energy decay, now only used to record refractive angle. Defaults to None.
            opl (Tensor, optional): Optical path length, Now used as the phase term. Defaults to None.
            coherent (bool, optional): If the ray is coherent. Defaults to False.
            device (torch.device, optional): Defaults to torch.device('cuda:0').
        """
        assert wvln > 0.1 and wvln < 1, 'wvln should be in [um]'
        self.wvln = wvln

        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.ra = torch.ones(o.shape[:-1])
        
        # not used
        self.en = torch.ones(o.shape[:-1])
        
        # used in coherent ray tracing
        self.coherent = coherent
        self.opl = torch.zeros(o.shape[:-1])
        self.phi = torch.zeros(o.shape[:-1])

        # used in lens design
        self.obliq = torch.ones(o.shape[:-1])  
                
        self.to(device)
        self.d = nnF.normalize(self.d, p=2, dim=-1)


    def prop_to(self, z, n=1):
        """ Ray propagates to a given depth. 
        """
        return self.propagate_to(z, n)


    def propagate_to(self, z, n=1):
        """ Ray propagates to a given depth.

            Args:
                z (float): depth.
                n (float, optional): refractive index. Defaults to 1.
        """
        o0 = self.o.clone()
        t = (z - self.o[..., 2]) / self.d[..., 2]
        self.o = self.o + self.d * t[..., None]
        
        if self.coherent:
            # ==> Update phase during propagation.
            assert self.wvln > 0.1 and self.wvln < 1, 'wvln should be in [um]'
            if t.min() > 100:
                # first propagation, long distance, in air
                opd = - (self.o * o0).sum(-1) / (o0 * o0).sum(-1).sqrt() 
                self.opl = self.opl + opd
            else:
                self.opl = self.opl + n * t # path length = t, because ||d|| = 1

        return self


    def project_to(self, z):
        """ Calculate the intersection points of ray with plane z.

            Return:
                p: shape of [..., 2].
        """
        t = (z - self.o[...,2]) / self.d[...,2]
        p = self.o[...,0:2] + self.d[...,0:2] * t[...,None]
        return p


    def clone(self, device=None):
        """ Clone a Ray object.
            
            Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        if device is None:
            return copy.deepcopy(self).to(self.device)
        else:
            return copy.deepcopy(self).to(device)


class Material():
    def __init__(self, name=None):
        self.name = 'vacuum' if name is None else name.lower()
        self.A, self.B = self._lookup_material()
        
        if self.name in SELLMEIER_TABLE:
            self.dispersion = 'sellmeier'
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = SELLMEIER_TABLE[self.name]
            self.glassname = self.name
        elif self.name in SCHOTT_TABLE:
            self.dispersion = 'schott'
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = SCHOTT_TABLE[self.name]
            self.glassname = GLASS_NAME[self.name]
        else:
            self.dispersion = 'naive'
            self.glassname = self.name

    def ior(self, wvln):
        """ Compute the refractive index at given wvln. 
            
            Reference: Zemax user manual.
        """
        wv = wvln if wvln < 10 else wvln * 1e-3 # use [um]
        
        # Compute refraction index
        if self.dispersion == 'sellmeier':
            # https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = 1 + self.k1 * wv**2 / (wv**2 - self.l1) + self.k2 * wv**2 / (wv**2 - self.l2) + self.k3 * wv**2 / (wv**2 - self.l3)
            n = np.sqrt(n2)
        elif self.dispersion == 'schott':
            # High precision computation (by MATLAB), writing dispersion function seperately will introduce errors 
            ws = wv**2
            n2 = self.a0 + self.a1*ws + (self.a2 + (self.a3 + (self.a4 + self.a5/ws)/ws)/ws)/ws
            n = np.sqrt(n2)
        elif self.dispersion == 'naive':
            # Use Cauchy function
            n = self.A + self.B / (wv * 1e3)**2

        return n

    def load_sellmeier_param(self, params=None):
        """ Manually set sellmeier parameters k1, l1, k2, l2, k3, l3.
            
            This function is called when we want to use a custom material.
        """
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params

    @staticmethod
    def nV_to_AB(n, V):
        """ Convert (n ,V) paramters to (A, B) parameters to find the material.
        """
        def ivs(a): return 1./a**2
        lambdas = [656.3, 589.3, 486.1]
        B = (n - 1) / V / ( ivs(lambdas[2]) - ivs(lambdas[0]) )
        A = n - B * ivs(lambdas[1])
        return A, B

    def _lookup_material(self):
        """ Find (A, B) parameters of the material. 
        
            (A, B) parameters are used to calculate the refractive index in the old implementation (Cauchy's equation).
        """
        out = MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            n, V = out
        elif out is None:
            # try parsing input as a n/V pair
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])

        self.n = n
        self.V = V
        return self.nV_to_AB(n, V)



# ----------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------
def wave_rgb():
    """ Randomly select one wave from R, G, B spectrum and return the wvln list (length 3)
    """
    wave_r = np.random.choice([0.620, 0.660, 0.700])
    wave_g = np.random.choice([0.500, 0.530, 0.560])
    wave_b = np.random.choice([0.450, 0.470, 0.490])
    return [wave_r, wave_g, wave_b]


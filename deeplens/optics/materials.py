""" Glass and plastic materials.
"""
import numpy as np
from .basics import *

MATERIAL_TABLE = { 
    # https://www.schott.com/en-dk/interactive-abbe-diagram
    # https://refractiveindex.info/

    # [nD, Abbe number]
    "vacuum":       [1.0000,   np.inf],
    "air":          [1.0000,   np.inf],
    "occluder":     [1.0000,   np.inf],
    
    "pmma":         [1.4918,   57.44],
    "n-pk52":       [1.4970,   81.63],
    "bk10":         [1.4978,   66.95],
    "bk1":          [1.5101,   63.47],
    "bk7":          [1.5168,   64.17],
    "n-bk7":        [1.5168,   64.17],
    "d-k59":        [1.5175,   63.50],
    "d-k59":        [1.5176,   63.50],
    "b270":         [1.5229,   58.50],
    "n-pk51":       [1.5310,   56.00],
    "coc":          [1.5337,   56.22],
    "s-til6":       [1.5317,   48.84],
    "apl5014cl":    [1.5445,   55.99],
    "hk51":         [1.5501,   58.64],
    "n-bak4":       [1.5688,   55.98],
    "n-balf4":      [1.5796,   53.86],
    "s-bal42":      [1.5831,   59.37],
    "polycarb":     [1.5855,   29.91],
    "pc":           [1.5855,   29.91],
    "d-zk3":        [1.5891,   61.15],
    "polystyr":     [1.5904,   30.87],
    "ps":           [1.5904,   30.87],
    "f5":           [1.6034,   38.03],
    "sk1":          [1.6102,   56.50],
    "sk1":          [1.6103,   56.71],
    "kzfs1":        [1.6131,   44.34],
    "ssk4":         [1.6177,   55.12],
    "f2":           [1.6200,   36.37],
    "flint":        [1.6200,   36.37],
    "sk16":         [1.6204,   60.31],
    "sk16":         [1.6204,   60.32],
    "n-ssk2":       [1.6223,   53.27],
    "okp4":         [1.6328,   23.34],
    "okp4ht":       [1.6328,   23.34],
    "n-baf10":      [1.6700,   47.11],
    "sf5":          [1.6727,   32.21],
    "h-zf2":        [1.6727,   32.17],
    "laf20":        [1.6825,   48.20],
    "h-lak51":      [1.6968,   55.53],
    "s-lal14":      [1.6968,   55.53],
    "s-tim35":      [1.6989,   30.13],
    "n-sf10":       [1.7283,   28.53],
    "lafn7":        [1.7495,   34.95],
    "n-lak34":      [1.7550,   52.30],
    "n-sf6":        [1.8052,   25.36],
    "s-nph1":       [1.8078,   22.76], 
    "s-lah55vs":    [1.8348,   42.74],
    "n-sf57":       [1.8467,   23.78],
    "sf59":         [1.8467,   23.83],
    "sf11":         [1.8745,   25.68],
    
}


SELLMEIER_TABLE = {
    # https://en.wikipedia.org/wiki/Sellmeier_equation
    # https://shop.schott.com/advanced_optics/
    # https://refractiveindex.info, nd at 589.3 [nm]

    # Sellmeier equation parameters.
    "air":          [0., 0., 0., 0., 0., 0.],
    "vacuum":       [0., 0., 0., 0., 0., 0.],
    "occluder":     [0., 0., 0., 0., 0., 0.],

    "bk1":          [1.0425, 6.1656e-3, 2.0838e-1,  2.1215e-2, 9.8014e-1,  1.0906e2],
    "bk7":          [1.0396, 6.0006e-3, 2.3179e-1,  2.0017e-2, 1.0104,     1.0356e2],
    "d-k59":        [1.1209, 6.5791e-3, 1.5269e-1,  2.3572e-2, 1.0750000,  1.0631e2],
    "d-zk3":        [1.3394, 0.0076061, 0.1486902,  0.0238444, 1.0095403,  89.04198],
    "f2":           [1.3453, 9.9774e-3, 2.0907e-1,  4.7045e-2, 9.3736e-1,  1.1188e2],
    "f5":           [1.3104, 9.5863e-3, 1.9603e-1,  4.5762e-2, 9.6612e-1,  1.1501e2],
    'hk51':         [0.9602, 116.24248, 1.1836896,  0.0118030, 0.1023382, -0.018958],
    "h-zf2":        [0.1676, 0.0605178, 1.5433507,  0.0118524, 1.1731312,  113.6711],
    "h-lak51":      [1.1875, 0.0158001, 0.6393986,  5.6713e-5, 1.2654535,  91.09705],
    "kzfs1":        [1.3661, 8.7316e-3, 1.8204e-1,  3.8983e-2, 8.6431e-1,  6.2939e1],
    "laf20":        [1.6510, 9.7050e-3, 1.1847e-1,  4.2892e-2, 1.1154,     1.1405e2],
    "lafn7":        [1.6684, 1.0316e-2, 2.9851e-1,  4.6922e-2, 1.0774,     8.2508e1],
    "n-balf4":      [1.3100, 7.9659e-3, 1.4208e-1,  3.3067e-2, 9.6493e-1,  1.0919e2],
    "n-bk7":        [1.0396, 6.0006e-3, 2.3179e-1,  2.0017e-2, 1.0104,     1.0356e2],
    "n-lak34":      [1.2666, 5.8928e-3, 6.6592e-1,  1.9751e-2, 1.1247,     78.88900],
    "n-pk51":       [1.1516, 5.8556e-3, 1.5323e-1,  1.9407e-2, 7.8562e-1,  140.5370],
    "n-pk52":       [1.0081, 5.0197e-3, 2.0943e-1,  1.6248e-2, 7.8169e-1,  1.5239e2],
    "n-sf6":        [1.7793, 0.0133714, 0.3381499,  0.0617533, 2.0873447,  174.0176],
    "s-bal42":      [1.3957, 0.0112219, 0.0718505, -0.0252117, 1.2712927,  134.4979],
    "sf11":         [1.7385, 1.3607e-2, 3.1117e-1,  6.1596e-2, 1.1749,     1.2192e2],
    "sf59":         [1.8165, 0.0143704, 0.4288936,  0.0592801, 1.0718628,  121.4199],
    "sk16":         [1.3431, 7.0468e-3, 2.4114e-1,  2.2900e-2, 9.9432e-1,  9.2751e1],
    "s-lah55vs":    [1.9259, 0.0096115, 0.3489535,  0.0365133, 1.4223074,  103.3641],
    "s-til6":       [1.1770, 0.0077109, 0.1279580,  0.0411325, 1.3474012,  154.5319],
    "s-lal14":      [1.2372, 0.0153551, 0.5897226, -0.0003079, 1.3192188,  93.72030],
    "s-tim35":      [1.5585, 0.0115367, 0.2307670,  0.0586096, 1.8443610,  162.9819],
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


INTERP_TABLE = {
    # Refractive indices from 0.4um to 0.7um for interpolation, 0.01um step size
    
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





class Material():
    def __init__(self, name=None):
        self.name = 'vacuum' if name is None else name.lower()
        self.load_dispersion()
        
    def load_dispersion(self):
        """ Load material dispersion equation.
        """
        if self.name in SELLMEIER_TABLE:
            self.dispersion = 'sellmeier'
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = SELLMEIER_TABLE[self.name]
            self.n, self.V = MATERIAL_TABLE[self.name]
        
        elif self.name in SCHOTT_TABLE:
            self.dispersion = 'schott'
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = SCHOTT_TABLE[self.name]
            self.n, self.V = MATERIAL_TABLE[self.name]
        
        elif self.name in MATERIAL_TABLE:
            self.dispersion = 'cauchy'
            self.n, self.V = MATERIAL_TABLE[self.name]
            self.A, self.B = self.nV_to_AB(self.n ,self.V)
        
        else:
            self.dispersion = 'cauchy'
            self.n, self.V = float(self.name.split('/')[0]), float(self.name.split('/')[1])
            self.A, self.B = self.nV_to_AB(self.n ,self.V)
        
    def ior(self, wvln):
        """ Compute the refractive index at given wvln. 
        """
        assert wvln > 0.1 and wvln < 1, 'Wavelength should be in [um].'
        
        if self.dispersion == 'sellmeier':
            # Sellmeier equation
            # https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = 1 + self.k1 * wvln**2 / (wvln**2 - self.l1) + self.k2 * wvln**2 / (wvln**2 - self.l2) + self.k3 * wvln**2 / (wvln**2 - self.l3)
            n = np.sqrt(n2)
        
        elif self.dispersion == 'schott':
            # High precision computation (by MATLAB), writing dispersion function seperately will introduce errors 
            ws = wvln**2
            n2 = self.a0 + self.a1*ws + (self.a2 + (self.a3 + (self.a4 + self.a5/ws)/ws)/ws)/ws
            n = np.sqrt(n2)
        
        elif self.dispersion == 'cauchy':
            # Cauchy equation
            # https://en.wikipedia.org/wiki/Cauchy%27s_equation 
            n = self.A + self.B / (wvln * 1e3)**2

        elif self.dispersion == 'optimizable':
            # Cauchy's equation, calculate (A, B) on the fly
            B = (self.n - 1) / self.V / (1 / 0.486**2 - 1 / 0.656**2)
            A = n - B * 1 / 0.589**2

            n = A + B / wvln**2

        else:
            raise NotImplementedError

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

    def match_material(self):
        """ Find the closest material in the database.
        """
        weight_n = 2
        dist_min = 1e6
        for name in MATERIAL_TABLE:
            n, V = MATERIAL_TABLE[name]
            dist = weight_n * abs(n - self.n) / self.n + abs(V - self.V) / self.V
            if dist < dist_min:
                self.name = name
                dist_min = dist
                
        self.load_dispersion()

    def get_optimizer_params(self, lr=1e-3):
        """ Optimize the material parameters (n, V).
        """
        self.n = torch.Tensor([self.n]).to(self.device)
        self.V = torch.Tensor([self.V]).to(self.device)

        self.n.requires_grad = True
        self.V.requires_grad = True
        self.dispersion = 'optimizable'

        params = {'params': [self.A, self.B], 'lr': lr}
        return params
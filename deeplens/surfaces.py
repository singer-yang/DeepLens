""" Lens surface class.

Technical Paper:
Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint (2023)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import torch
import numpy as np
import torch.nn.functional as nnF

from .basics import *
from .utils import *

class Surface(DeepObj):
    def __init__(self, r, d, mat1, mat2, is_square=False, device=DEVICE):
        super(Surface, self).__init__()
        self.d = d if torch.is_tensor(d) else torch.Tensor([d])
        self.r = float(r)   # r is not differentiable
        self.is_square = is_square
        if is_square:
            self.h = r * np.sqrt(2)
            self.w = r * np.sqrt(2)

        self.mat1 = Material(mat1)
        self.mat2 = Material(mat2)        

        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 10e-6 # in [mm], here is 10 [nm] 
        self.NEWTONS_TOLERANCE_LOOSE = 50e-6 # in [mm], here is 50 [nm] 
        self.NEWTONS_STEP_BOUND = 5 # [mm], maximum iteration step in Newton's iteration
        self.APERTURE_SAMPLING = 257

        self.to(device)

    # ==============================
    # Intersection and Refraction
    # ==============================
    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.
        """
        # Determine ray direction and refractive index
        wvln = ray.wvln
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            n1 = self.mat1.ior(wvln)
            n2 = self.mat2.ior(wvln)
        else:
            n1 = self.mat2.ior(wvln)
            n2 = self.mat1.ior(wvln)

        # Intersection
        ray = self._intersect(ray, n1)

        # Refraction
        ray = self._refract(ray, n1 / n2)

        return ray
    
    def _intersect(self, ray, n = 1.0):
        """ Solve ray-surface intersection and update ray data.
        """
        # Solve intersection time t by Newton's method
        t, valid = self._newtons_method(ray)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            assert t.min() < 100, 'Precision problem caused by long propagation distance.'
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def _newtons_method(self, ray):
        """ Solve intersection by Newton's method. This function will only update valid rays.
        """
        # 1. inital guess of t
        t0 = (self.d - ray.o[...,2]) / ray.d[...,2]   # if the shape of aspheric surface is strange, will hit the back surface region instead 

        # 2. use Newton's method to update t to find the intersection points (non-differentiable)
        with torch.no_grad():
            it = 0
            t = t0  # initial guess of t
            ft = MAXT * torch.ones_like(ray.o[...,2])
            while (torch.abs(ft) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < self.NEWTONS_MAXITER):
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[...,0], new_o[...,1]
                valid = self._valid(new_x, new_y) & (ray.ra>0)
                
                ft = self.sag(new_x, new_y, valid) + self.d - new_o[...,2]
                dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

            t1 = t - t0

        # 3. do one more Newton iteration to gain gradients
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[...,0], new_o[...,1]
        valid = self._valid(new_x, new_y) & (ray.ra > 0)
        
        ft = self.sag(new_x, new_y, valid) + self.d - new_o[...,2]
        dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
        dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
        dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
        t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

        # determine valid rays
        with torch.no_grad():
            new_x, new_y = new_o[...,0], new_o[...,1]
            valid = self._valid_within_boundary(new_x, new_y) & (ray.ra > 0)
            ft = self.sag(new_x, new_y, valid) + self.d - new_o[...,2]
            valid = valid & (torch.abs(ft.detach()) < self.NEWTONS_TOLERANCE_TIGHT) & (t > 0)   # points valid & points accurate & donot go back
        
        return t, valid


    def _refract(self, ray, eta):
        """ Snell's law (surface normal n defined along the positive z axis)
            https://physics.stackexchange.com/a/436252/104805
            https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

            We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it
            points to left. We use the second link to check.

            veci: incident ray
            vect: refractive ray
            eta: relevant refraction coefficient, eta = eta_i/eta_t
        """
        # Compute normal vectors
        n = self._normal(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            n = - n

        # Compute refraction according to Snell's law
        cosi = torch.sum(ray.d * n, axis=-1)   # n * i

        # TIR
        valid = (eta**2 * (1 - cosi**2) < 1) & (ray.ra > 0)

        sr = torch.sqrt(1 - eta**2 * (1 - cosi.unsqueeze(-1)**2) * valid.unsqueeze(-1))  # square root
        
        # First term: vertical. Second term: parallel. Already normalized if both n and ray.d are normalized. 
        new_d = sr * n + eta * (ray.d - cosi.unsqueeze(-1) * n)
        new_d[~valid] = ray.d[~valid]
        
        # Update valid rays
        ray.d = new_d
        ray.ra = ray.ra * valid

        return ray


    def _normal(self, ray):
        """ Calculate normal vector of the surface at intersection point.
        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        nx, ny, nz = self.dfdxyz(x, y)
        n = torch.stack((nx, ny, nz), axis = -1)
        n = nnF.normalize(n, p = 2, dim = -1)

        return n
    
    # =================================================================================
    # Calculation-related methods
    # =================================================================================
    def sag(self, x, y, valid=None):
        """ Calculate sag (z) of the surface. z = f(x, y)

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        if valid is None:
            valid = self._valid(x, y)
        
        x, y = x * valid, y * valid
        return self.g(x, y)

    def dfdxyz(self, x, y, valid=None):
        """ Compute derivatives of surface function. Surface function: f(x, y, z): z - g(x, y) = 0

            NOTE: this function only works for surfaces which can be written as z = g(x, y). For implicit surfaces, we need to compute derivatives (df/dx, df/dy, df/dz).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            dfdx (tensor): df / dx
            dfdy (tensor): df / dy
            dfdz (tensor): df / dz
        """
        if valid is None:
            valid = self._valid(x, y)
        
        x, y = x * valid, y * valid
        dx, dy = self.dgd(x, y)
        return dx, dy, - torch.ones_like(x)
    
    def g(self, x, y):
        """ Calculate sag (z) of the surface. z = f(x, y)

            NOTE: Valid term is used to avoid NaN when x, y are super large, which happens in spherical and aspherical surfaces. But if you want to calculate r = sqrt(x**2, y**2), this will cause another NaN error when calculating dr/dx = x / sqrt(x**2 + y**2). So be careful for this!!!

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        raise NotImplementedError()
        
    def dgd(self, x, y):
        """ Compute derivatives of sag to x and y. (dgdx, dgdy) =  (g'x, g'y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            
        Return:
            dgdx (tensor): dg / dx
            dgdy (tensor): dg / dy
        """
        raise NotImplementedError()
    
    def is_valid(self, p):
        return (self.sdf_approx(p) < 0.0).bool()

    def _valid_within_boundary(self, x, y):
        """ Valid points within the boundary of the surface.
        """
        if self.is_square:
            valid = self._valid(x, y) & (torch.abs(x) <= self.w/2) & (torch.abs(y) <= self.h/2)
        else:
            valid = self._valid(x, y) & ((x**2 + y**2) <= self.r**2)
        
        return valid
    
    def _valid(self, x, y):
        """ Valid points NOT considering the boundary of the surface.
        """
        return torch.ones_like(x, dtype=torch.bool)
    
    def surface_sample(self, N=1000):
        """ Sample uniform points on the surface.
        """
        r_max = self.r
        theta = torch.rand(N)*2*np.pi
        r = torch.sqrt(torch.rand(N)*r_max**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, self.d.item())
        o2 = torch.stack((x2,y2,z2), 1).to(self.device)
        return o2
    
    def surface_with_offset(self, x, y):
        """ Calculate z coordinate of the surface at (x, y) with offset.
            
            This function is used in lens setup plotting.
        """
        x = torch.tensor(x).to(self.device) if type(x) is float else x
        y = torch.tensor(y).to(self.device) if type(y) is float else y
        return self.sag(x, y) + self.d
    
    def max_height(self):
        """ Maximum valid height.
        """
        raise NotImplementedError()
    
    # =========================================
    # Optimization-related methods
    # =========================================
    def activate_grad(self, activate=True):
        raise NotImplementedError()
    
    def get_optimizer_params(self, lr):
        raise NotImplementedError()

    def get_optimizer(self, lr):
        params = self.get_optimizer_params(lr)
        return torch.optim.Adam(params)

    def surf_dict(self):
        surf_dict = {
            'type': self.__class__.__name__,
            'r': self.r,
            'd': self.d,
            'is_square': self.is_square,
            'mat1': self.mat1.name,
            'mat2': self.mat2.name,
        }

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        raise NotImplementedError()
    
    @torch.no_grad()
    def perturb(self, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors.
        """
        raise Exception('This function needs to be implemented in the child class.')
        self.r += np.random.randn() * diameter_precision
        self.d += torch.randn() * thickness_precision


class Aperture(Surface):
    def __init__(self, r, d, diffraction=False, device=DEVICE):
        """ Aperture, can be circle or rectangle. 
            For geo optics, it works as a binary mask.
            For wave optics, it works as a diffractive plane.
        """
        Surface.__init__(self, r, d, mat1='air', mat2='air', is_square=False, device=device)
        self.diffraction = diffraction
        self.to(device)

    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.

            In each step, first get a guess of new o and d, then compute valid and only update valid rays. 
        """
        # -------------------------------------
        # Intersection
        # ------------------------------------- 
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) <= self.r) & (ray.ra > 0)

        # => Update position
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        # => Update phase
        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        # -------------------------------------
        # Diffraction
        # ------------------------------------- 
        if self.diffraction:
            raise Exception('Unimplemented diffraction method.')
            # We can use Huygens-Fresnel principle to determine the diffractive rays, but how to make this process differentiable???? Using Heisenberg uncertainty principle???
            # Ref: Simulating multiple diffraction in imaging systems using a path integration method
            # Conventional method process the aperture by using the exit pupil + free space propagation, if that we donot need this class.

        return ray
    
    def surf_dict(self):
        """ Return a dict of surface.
        """
        surf_dict = {
            'type': 'Aperture',
            'r': self.r,
            'd': self.d.item(),
            'is_square': self.is_square,
            'diffraction': self.diffraction,
            'mat1': 'air',
            'mat2': 'air',
            }
        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
"""
        return zmx_str

class Aspheric(Surface):
    """ This class can represent plane, spheric and aspheric surfaces.

        Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.

        Three kinds of surfaces:
            1. flat: always use round 
            2. spheric: 
            3. aspheric: 
    """
    def __init__(self, r, d, c=0., k=0., ai=None, mat1=None, mat2=None, device=DEVICE):
        """ Initialize aspheric surface.

        Args:
            r (float): radius of the surface
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): aspheric coefficients
            mat1 (Material): material of the first medium
            mat2 (Material): material of the second medium
            is_square (bool): whether the surface is square
            device (torch.device): device to store the tensor
        """
        Surface.__init__(self, r, d, mat1, mat2, is_square=False, device=device)
        self.c = torch.Tensor([c])
        self.k = torch.Tensor([k])
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[2]])
            elif self.ai_degree == 5:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
                self.ai10 = torch.Tensor([ai[4]])
            elif self.ai_degree == 6:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
            else:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
        else:
            self.ai = None
            self.ai_degree = 0
        
        self.to(device)


    def init(self, ai_degree=6):
        """ Initialize all parameters.
        """
        self.init_c()
        self.init_k()
        self.init_ai(ai_degree=ai_degree)
        self.init_d()


    def init_c(self, c_bound=0.0002):
        """ Initialize lens surface c parameters by small values between [-0.05, 0.05], 
            which means roc should be (-inf, 20) or (20, inf)
        """
        self.c = c_bound * (torch.rand(1) - 0.5).to(self.device)

    def init_ai(self, ai_degree=3, bound=0.0001):
        """ If ai is None, set to random value.
            For different length, create a new initilized value and set original ai.
        """
        old_ai_degree = self.ai_degree
        self.ai_degree = ai_degree
        if old_ai_degree == 0:
            if ai_degree == 4:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
            elif ai_degree == 5:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
                self.ai10 = (torch.rand(1, device=self.device)-0.5) * bound* 0.001
            elif ai_degree == 6:
                for i in range(1, self.ai_degree+1):
                    exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')
            else:
                raise Exception('Wrong ai degree')
        else:
            for i in range(old_ai_degree + 1, self.ai_degree + 1):
                exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')

    
    def init_k(self, bound=1):
        """ When k is 0, set to a random value.
        """
        if self.k == 0:
            k = torch.rand(1) * bound
            self.k = k.to(self.device) 


    def init_d(self, bound = 0.1):
        return

    
    def g(self, x, y):
        """ Compute surface height.
        """
        r2 = x**2 + y**2
        total_surface = r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2))

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4
            elif self.ai_degree == 5:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5
            elif self.ai_degree == 6:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5 + self.ai12 * r2 ** 6
            elif self.ai_degree == 7:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + self.ai14 * r2) * r2) * r2) * r2) * r2) * r2) * r2
            elif self.ai_degree == 8:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + (self.ai14 + self.ai16 * r2)* r2) * r2) * r2) * r2) * r2) * r2) * r2
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'total_surface += self.ai{2*i} * r2 ** {i}')

        return total_surface


    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2)
        dsdr2 = (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf)**2

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3
            elif self.ai_degree == 5:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4
            elif self.ai_degree == 6:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4 + 6 * self.ai12 * r2 ** 5
            elif self.ai_degree == 8:
                dsdr2 = dsdr2 + self.ai2 + (2 * self.ai4 + (3 * self.ai6 + (4 * self.ai8 + (5 * self.ai10 + (6 * self.ai12 + (7 * self.ai14 + 8 * self.ai16 * r2)* r2)* r2)* r2) * r2) * r2) * r2 
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'dsdr2 += {i} * self.ai{2*i} * r2 ** {i-1}')

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def _valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        if self.k > -1:
            valid = ((x**2 + y**2) < 1 / self.c**2 / (1 + self.k))
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """ Maximum valid height.
        """
        if self.k > -1:
            max_height = torch.sqrt(1 / (self.k + 1) / (self.c**2)).item() - 0.01
        else:
            max_height = 100

        return max_height

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-4], decay=0.1):
        """ Get optimizer parameters for different parameters.

        Args:
            lr (list, optional): learning rates for c, d, k, ai. Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr*1e3, lr]

        params = []
        if lr[0] > 0 and self.c != 0:
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr[0]})
        if lr[1] > 0:
            self.d.requires_grad_(True)
            params.append({'params': [self.d], 'lr': lr[1]})
        if lr[2] > 0 and self.k != 0:
            params.append({'params': [self.k], 'lr': lr[2]})
        if lr[3] > 0:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3] / decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**2})
            if self.ai_degree == 5:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3] / decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**2})
                params.append({'params': [self.ai10], 'lr': lr[3] * decay**3})
            if self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3] / decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**2})
                params.append({'params': [self.ai10], 'lr': lr[3] * decay**3})
                params.append({'params': [self.ai12], 'lr': lr[3] * decay**4})
            else:
                for i in range(2, self.ai_degree + 1):
                    exec(f'self.ai{2*i}.requires_grad_(True)')
                    exec(f'params.append({{\'params\': [self.ai{2*i}], \'lr\': lr[3] / decay**{i-1}}})')
        
        return params


    @torch.no_grad()
    def perturb(self, ratio=0.001, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors. This function should only be called in the final image simulation stage. 
        
        Args:
            ratio (float, optional): perturbation ratio. Defaults to 0.001.
            thickness_precision (float, optional): thickness precision. Defaults to 0.0005.
            diameter_precision (float, optional): diameter precision. Defaults to 0.001.
        """
        self.r += np.random.randn() * diameter_precision
        if self.c != 0:
            self.c *= 1 + np.random.randn() * ratio
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision
        if self.k != 0:
            self.k *= 1 + np.random.randn() * ratio
        for i in range(1, self.ai_degree+1):
            exec(f'self.ai{2*i} *= 1 + np.random.randn() * ratio')


    def surf_dict(self):
        """ Return a dict of surface.
        """
        surf_dict = {
            'type': 'Aspheric',
            'r': self.r,
            'c': self.c.item(),
            'roc': 1 / self.c.item(),
            'd': self.d.item(),
            'k': self.k.item(),
            'ai': [],
            'mat1': self.mat1.name,
            'mat2': self.mat2.name,
            }
        for i in range(1, self.ai_degree+1):
            exec(f'surf_dict[\'ai{2*i}\'] = self.ai{2*i}.item()')
            surf_dict['ai'].append(eval(f'self.ai{2*i}.item()'))

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        assert self.c.item() != 0, 'Aperture surface is re-implemented in Aperture class.'
        assert self.ai is not None or self.k != 0, 'Spheric surface is re-implemented in Spheric class.'
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {self.d.item()}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        return zmx_str


class Cubic(Surface):
    """ Cubic surface: z(x,y) = b3 * (x**3 + y**3)

        Actually Cubic phase is a group of surfaces with changing height.

        Can also be written as: f(x, y, z) = 0
    """
    def __init__(self, r, d, ai, mat1, mat2, is_square=False, device=DEVICE):
        Surface.__init__(self, r, d, mat1, mat2, is_square=is_square, device=device) 
        self.ai = torch.Tensor(ai)

        if len(ai) == 1:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b_degree = 1
        elif len(ai) == 2:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b_degree = 2
        elif len(ai) == 3:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b7 = torch.Tensor([ai[2]]).to(device)
            self.b_degree = 3
        else:
            raise Exception('Unsupported cubic degree!!')

        self.rotate_angle = 0.0

    def g(self, x, y):
        """ Compute surface height z(x, y).
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5) + self.b7 * (x**7 + y**7)
        else:
            raise Exception('Unsupported cubic degre!')
        
        if len(z.size()) == 0:
            z = torch.Tensor([z]).to(self.device)

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)
        
        return z

    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise Exception('Unsupported cubic degree!')

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        return sx, sy

    def get_optimizer_params(self, lr):
        """ Return parameters for optimizer.
        """
        params = []
        
        self.d.requires_grad_(True)
        params.append({'params': [self.d], 'lr': lr})
        
        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
            params.append({'params': [self.b5], 'lr': lr * 0.1})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
            params.append({'params': [self.b5], 'lr': lr * 0.1})
            params.append({'params': [self.b7], 'lr': lr * 0.01})
        else:
            raise Exception('Unsupported cubic degree!')
        
        return params

    def perturb(self, curvature_precision=0.001, thickness_precision=0.0005, diameter_precision=0.01, angle=0.01):
        """ Perturb the surface
        """
        self.r += np.random.randn() * diameter_precision
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision

        if self.b_degree == 1:
            self.b3 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 2:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 3:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
            self.b7 *= 1 + np.random.randn() * curvature_precision

        self.rotate_angle = np.random.randn() * angle

    def surf_dict(self):
        """ Return surface parameters.
        """
        return {
            'type': 'cubic',
            'b3': self.b3,
            'b5': self.b5,
            'b7': self.b7,
            'r': self.r,
            'd': self.d,
            'rotate_angle': self.rotate_angle
        }


class Plane(Surface):
    def __init__(self, l, d, mat1, mat2, is_square=True, device=DEVICE):
        """ Plane surface, typically rectangle. Working as IR filter, lens cover or DOE base.
        """
        Surface.__init__(self, l / np.sqrt(2), d, mat1=mat1, mat2=mat2, is_square=is_square, device=device)
        
    def _intersect(self, ray, n=1.0):
        """ Solve ray-surface intersection and update ray data.
        """
        # Solve intersection
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (torch.abs(new_o[...,0]) < self.w/2) & (torch.abs(new_o[...,1]) < self.h/2) & (ray.ra > 0)
        else:
            valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) < self.r) & (ray.ra > 0)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def _normal(self, ray):
        """ Calculate surface normal vector at intersection points.
        """
        n = torch.zeros_like(ray.d)
        n[...,2] = -1
        return n

    def g(x, y):
        return torch.zeros_like(x)
    
    def dgd(x, y):
        return torch.zeros_like(x), torch.zeros_like(x)
    
    def surface(self, x, y):
        if torch.is_tensor(x):
            return torch.zeros_like(x)
        else:
            return 0

class Spheric(Surface):
    """ Spheric surface.
    """
    def __init__(self, c, r, d, mat1, mat2, device=DEVICE):
        super(Spheric, self).__init__(r, d, mat1, mat2, is_square=False, device=device)
        self.c = torch.tensor([c])
        self.to(device)

    def g(self, x, y):
        """ Compute surfaces sag z = r**2 * c / (1 - sqrt(1 - r**2 * c**2))
        """
        r2 = x**2 + y**2
        sag = self.c * r2 / (1 + torch.sqrt(1 - r2 * self.c**2))
        return sag

    def dgd(self, x, y):
        """ Compute surface sag derivatives to x and y: dz / dx, dz / dy.
        """
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * self.c**2)
        dgdr2 = (1 + sf + r2 * self.c**2 / 2 / sf) * self.c / (1 + sf)**2
        return dgdr2*2*x, dgdr2*2*y

    def _valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        valid = (x**2 + y**2 < 1 / self.c**2)
        return valid
    
    def max_height(self):
        """ Maximum valid height.
        """
        max_height = torch.sqrt(1 / self.c**2).item() - 0.01
        return max_height
    
    def get_optimizer_params(self, lr=[0.001, 0.001]):
        """ Activate gradient computation for c and d and return optimizer parameters.
        """
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)
        
        params = []
        params.append({'params': [self.c], 'lr': lr[0]})
        params.append({'params': [self.d], 'lr': lr[1]})
        return params

    def surf_dict(self):
        surf_dict = {
                'type': 'Spheric',
                'r': self.r,
                'c': self.c.item(),
                'roc': 1/self.c.item(),
                'd': self.d.item(),
                'mat1': self.mat1.name,
                'mat2': self.mat2.name,
                }

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r*2}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r*2}
"""

        return zmx_str

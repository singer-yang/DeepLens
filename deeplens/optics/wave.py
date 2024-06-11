""" Complex wave class. We have to use float64 precision.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.fft import *
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
import pickle
from torchvision.utils import save_image

from .basics import *

# ===================================
# Complex wave field
# ===================================
class ComplexWave(DeepObj):
    def __init__(self, u=None, wvln=0.550, z=0., phy_size=[4., 4.], valid_phy_size=None, res=[1024,1024], device=DEVICE):
        """ Complex wave field class.

        Args:
            amp (_type_, optional): _description_. Defaults to None.
            phi (_type_, optional): _description_. Defaults to None.
            wvln (int, optional): wvln. Defaults to 550.
            size (list, optional): physical size of a field, in [mm].
            res (list, optional): discrete resolution of a field. Defaults to [128,128].

            res
            phy_size
            wvln: [um]
            k
            
            u: can be either [B, 1, H, W] or [H, W]
            x: [H, W]
            y: [H, W]
            z: [H, W]
        """
        super(ComplexWave, self).__init__()

        # Wave field has shape of [N, 1, H, W] for batch processing
        if u is not None:
            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)
            
            if len(u.shape) == 2:   # [H, W]
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:    # [1, H, W]
                self.u = self.u.unsqueeze(0)
            
            self.res = self.u.shape[-2:]

        else:
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res        

        # Other paramters
        assert wvln > 0.1 and wvln < 1, 'wvln unit should be [um].'
        self.wvln = wvln # wvln, store in [um]
        self.k = 2 * np.pi / (self.wvln * 1e-3) # distance unit [mm]
        self.phy_size = np.array(phy_size)  # physical size with padding, in [mm]
        self.valid_phy_size = self.phy_size if valid_phy_size is None else np.array(valid_phy_size) # physical size without padding, in [mm]
        
        assert phy_size[0] / self.res[0] == phy_size[1] / self.res[1], 'Wrong pixel size.'
        self.ps = phy_size[0] / self.res[0] # pixel size, float value

        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(self.x, z)

        self.to(device)
        

    def load_img(self, img):
        """ Use the pixel value of an image/batch as the amplitute.

        Args:
            img (ndarray or tensor): shape [H, W] or [B, C, H, W].
        """
        if img.dtype == 'uint8':
            img = img/255.

        if torch.is_tensor(img):
            amp = torch.sqrt(img)
        else:
            amp = torch.sqrt(torch.from_numpy(img/255.))
        
        phi = torch.zeros_like(amp)
        u = amp + 1j * phi
        self.u = u.to(self.device)
        self.res = self.u.shape


    def load_pkl(self, data_path):
        with open(data_path, "rb") as tf:
            wave_data = pickle.load(tf)
            tf.close()

        amp = wave_data['amp']
        phi = wave_data['phi']
        self.u = amp * torch.exp(1j * phi)
        self.x = wave_data['x']
        self.y = wave_data['y']
        self.wvln = wave_data['wvln']
        self.phy_size = wave_data['phy_size']
        self.valid_phy_size = wave_data['valid_phy_size']
        self.res = self.x.shape

        self.to(self.device)


    def save_data(self, save_path='./test.pkl'):
        data = {
            'amp': self.u.cpu().abs(),
            'phi': torch.angle(self.u.cpu()),
            'x': self.x.cpu(),
            'y': self.y.cpu(),
            'wvln': self.wvln,
            'phy_size': self.phy_size,
            'valid_phy_size': self.valid_phy_size
            }

        with open(save_path, 'wb') as tf:
            pickle.dump(data, tf)
            tf.close()

        # cv.imwrite(f'{save_path[:-4]}.png', self.u.cpu().abs()**2)
        intensity = self.u.cpu().abs()**2
        save_image(intensity, f'{save_path[:-4]}.png', normalize=True)
        save_image(torch.abs(self.u.cpu()), f'{save_path[:-4]}_amp.jpg', normalize=True)
        save_image(torch.angle(self.u.cpu()), f'{save_path[:-4]}_phase.jpg', normalize=True)


    # =============================================
    # Operation
    # =============================================
    
    def flip(self):
        self.u = torch.flip(self.u, [-1,-2])
        self.x = torch.flip(self.x, [-1,-2])
        self.y = torch.flip(self.y, [-1,-2])
        self.z = torch.flip(self.z, [-1,-2])

        return self

    def prop(self, prop_dist, n = 1.):
        """ Propagate the field by distance z. Can only propagate planar wave. 

            The definition of near-field and far-field depends on the specific problem we want to solve. For diffraction simulation, typically we use Fresnel number to determine the propagation method. In Electro-magnetic applications and fiber optics, the definition is different.
        
            This function now supports batch operation, but only for mono-channel field. Shape of [B, 1, H, W].

            Reference: 
                1, https://spie.org/samples/PM103.pdf
                2, "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"

            Different methods:
                1, Rayleigh-Sommerfeld Diffraction Formula
                    pros: (a) intermediate and short distance, (b) non-paraxial, (c) ...
                    cons: (a) complexity, (b) scalar wave only, (c) not suitable for long distance, (d) ...
                2, Fresnel diffraction
                3, Fraunhofer diffraction
                4, Finite Difference Time Domain (FDTD)
                5, Beam Propagation Method (BPM)
                6, Angular Spectrum Method (ASM)
                7, Green's function method
                8, Split-step Fourier method

        Args:
            z (float): propagation distance, unit [mm].
        """
        wvln = self.wvln * 1e-3 # [um] -> [mm]
        valid_phy_size = self.valid_phy_size
        if torch.is_tensor(prop_dist):
            prop_dist = prop_dist.item()

        # Determine which propagation method to use by Fresnel number
        num_fresnel = valid_phy_size[0] * valid_phy_size[1] / (wvln * np.abs(prop_dist)) if prop_dist != 0 else 0
        if prop_dist < DELTA:
            # Zero distance, do nothing
            pass

        elif prop_dist < wvln / 2:
            # Sub-wavelength distance: EM method
            raise Exception('EM method is not implemented.')
        
        else:
            # Super short distance: Angular Spectrum Method
            prop_dist_min = self.Nyquist_zmin()
            if np.abs(prop_dist) < prop_dist_min:
                print("Propagation distance is too short.")
            self.u = AngularSpectrumMethod(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        self.z += prop_dist
        return self


    def prop_to(self, z, n=1):
        """ Propagate the field to plane z.

        Args:
            z (float): destination plane z coordinate.
        """
        if torch.is_tensor(z):
            z = z.item()
        prop_dist = z - self.z[0, 0].item()

        self.prop(prop_dist, n=n)
        return self

    def gen_xy_grid(self):
        """ To align with the image: Img[i, j] -> [x[i, j], y[i, j]]. Use top-left corner to represent the pixel.

            New: use the center of the pixel to represent the pixel.
        """
        ps = self.ps
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[0] + 0.5 * ps, 0.5 * self.phy_size[1] - 0.5 * ps, self.res[0]),
            torch.linspace(0.5 * self.phy_size[1] - 0.5 * ps, -0.5 * self.phy_size[0] + 0.5 * ps, self.res[1]), 
            indexing='xy')
        return x, y


    def gen_freq_grid(self):
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy


    def show(self, data='irr', save_name=None):
        """ Show the field.
        
        TODO: use x, y coordinates

        Args:
            data (str, optional): _description_. Defaults to 'irr'.

        Raises:
            Exception: _description_
            Exception: _description_
        """
        if data == 'irr':
            value = self.u.detach().abs()**2
            cmap = 'gray'
        elif data == 'amp':
            value = self.u.detach().abs()
            cmap = 'gray'
        elif data == 'phi' or data == 'phase':
            value = torch.angle(self.u).detach()
            cmap = 'hsv'
        elif data == 'real':
            value = self.u.real.detach()
            cmap = 'gray'
        elif data == 'imag':
            value = self.u.imag.detach()
            cmap = 'gray'
        else:
            raise Exception('Unimplemented visualization.')

        if len(self.u.shape) == 2:
            if save_name is not None:
                save_image(value, save_name, normalize=True)
            else:
                value = value.cpu().numpy()
                plt.imshow(value, cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
        
        elif len(self.u.shape) == 4:
            B, C, H, W = self.u.shape
            if B == 1:
                if save_name is not None:
                    save_image(value, save_name, normalize=True)
                else:
                    value = value.cpu().numpy()
                    plt.imshow(value[0, 0, :, :], cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
            else:
                if save_name is not None:
                    plt.savefig(save_name)
                else:
                    value = value.cpu().numpy()
                    fig, axs = plt.subplots(1, B)
                    for i in range(B):
                        axs[i].imshow(value[i, 0, :, :], cmap=cmap, extent=[-self.phy_size[0]/2, self.phy_size[0]/2, -self.phy_size[1]/2, self.phy_size[1]/2])
                    fig.show()
        else:
            raise Exception('Unsupported complex field shape.')
            
    def Nyquist_zmin(self):
        """ Compute Nyquist zmin, suppose the second plane has the same side length with the original plane.
        """
        wvln = self.wvln * 1e-3 # [um] to [mm]
        zmin = np.sqrt((4 * self.ps**2 / wvln**2 - 1) * (self.phy_size[0]/2 + self.phy_size[0]/2)**2)
        return zmin

    def pad(self, Hpad, Wpad):
        """ Pad the input field by (Hpad, Hpad, Wpad, Wpad). 
            This step will also expand the field.

            NOTE: Can only pad plane field.

        Args:
            Hpad (_type_): _description_
            Wpad (_type_): _description_
        """
        device = self.device

        # Pad directly
        self.u = nnf.pad(self.u, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=0)
        
        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [self.phy_size[0] * self.res[0] / Horg, self.phy_size[1] * self.res[1] / Worg]
        self.x, self.y = self.gen_xy_grid()
        z = self.z[0, 0]
        self.z = nnf.pad(self.z, (Hpad,Hpad,Wpad,Wpad), mode='constant', value=z)


# ===================================
# Diffraction functions
# ===================================
def AngularSpectrumMethod(u, z, wvln, ps, n=1., padding=True, TF=True):
    """ Rayleigh-Sommerfield propagation with FFT.

    Considerations:
        1, sampling requirement
        2, paraxial approximation
        3, boundary effects

        https://blog.csdn.net/zhenpixiaoyang/article/details/111569495
    
    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        wvln: wvln
        res: field resolution
        ps (float): pixel size
        z (float): propagation distance
    """
    if torch.is_tensor(z):
        z = z.item()

    # Reshape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)
    
    # Padding 
    if padding:
        Wpad, Hpad = Worg//2, Horg//2
        Wimg, Himg = Worg + 2*Wpad, Horg + 2*Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Wimg, Himg = Worg, Horg

    # Propagation
    assert wvln > 0.1 and wvln < 1, 'wvln unit should be [um].'
    k = 2 * np.pi / (wvln * 1e-3)   # we use k in vaccum, k in [mm]-1
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg, device=u.device),
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg, device=u.device),
        indexing='xy'
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5/ps, 0.5/ps, Wimg, device=u.device),
        torch.linspace(0.5/ps, -0.5/ps, Himg, device=u.device),
        indexing='xy'
    )

    # Determine TF or IR
    if ps > wvln * np.abs(z) / (Wimg * ps):
        TF = True
    else:
        TF = False

    if TF: 
        if n == 1:
            square_root = torch.sqrt(1 - (wvln*1e-3)**2 * (fx**2 + fy**2))
            H = torch.exp(1j * k * z * square_root)
        else:
            square_root = torch.sqrt(n**2 - (wvln*1e-3)**2 * (fx**2 + fy**2))
            H = n * torch.exp(1j * k * z * square_root)
        
        H = fftshift(H)
    
    else:
        r2 = x**2 + y**2 + z**2
        r = torch.sqrt(r2)

        if n == 1:
            h = z / (1j * wvln * r2) * torch.exp(1j * k * r)
        else:
            h = z * n / (1j * wvln * r2) * torch.exp(1j * n * k * r)
        
        H = fft2(fftshift(h)) * ps**2
    

    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    del x, y, fx, fy
    return u


def FresnelDiffraction(u, z, wvln, ps, n=1., padding=True, TF=None):
    """ Fresnel propagation with FFT.

    Ref: Computational fourier optics : a MATLAB tutorial
         https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m

    Args:
        u (_type_): _description_
        wvln (_type_): _description_
        ps (_type_): _description_
        z (_type_): _description_
        n (_type_, optional): _description_. Defaults to 1..
        padding (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # padding 
    if padding:
        try:
            _, _, Worg, Horg = u.shape
        except:
            Horg, Worg = u.shape
        Wpad, Hpad = Worg//2, Horg//2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        _, _, Wimg, Himg = u.shape

    # compute H function
    assert wvln < 10, 'wvln should be in [um].'
    k = 2 * np.pi / wvln
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg+1, device=u.device)[:-1],
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg+1, device=u.device)[:-1],
        indexing='xy'
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5/ps, 0.5/ps, Wimg+1, device=u.device)[:-1],
        torch.linspace(-0.5/ps, 0.5/ps, Himg+1, device=u.device)[:-1],
        indexing='xy'
    )
    # fx, fy = x/ps, y/ps

    # Determine TF or IR
    if TF is None:
        if ps > wvln * np.abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False
    # TF = True
    
    # Computational fourier optics. Chapter 5, section 5.1.
    if TF:
        # Correct, checked.
        if n == 1:
            H = torch.exp(- 1j * np.pi * wvln * z * (fx**2 + fy**2))
        else:
            H = np.sqrt(n) * torch.exp(- 1j * np.pi * wvln * z * (fx**2 + fy**2) / n)

        H = fftshift(H)
    else:
        if n == 1:
            h = 1 / (1j * wvln * z) * torch.exp(1j * k / (2*z) * (x**2+y**2)) 
        else:
            h = n / (1j * wvln * z) * torch.exp(1j * n * k / (2*z) * (x**2+y**2))
        
        H = fft2(fftshift(h)) * ps**2
    
    
    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # remove padding
    if padding:
        u = u[...,Wpad:-Wpad,Hpad:-Hpad]

    return u


def FraunhoferDiffraction(u, z, wvln, ps, n=1., padding=True):
    """ Fraunhofer propagation. 
    """
    # padding 
    if padding:
        Worg, Horg = u.shape
        Wpad, Hpad = Worg//4, Horg//4
        Wimg, Himg = Worg + 2*Wpad, Horg + 2*Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Wimg, Himg = u.shape

    # side length
    L2 = wvln * z / ps
    ps2 = wvln * z / Wimg / ps
    x2, y2 = torch.meshgrid(
        torch.linspace(- L2 / 2, L2 / 2, Wimg+1, device=u.device)[:-1],
        torch.linspace(- L2 / 2, L2 / 2, Himg+1, device=u.device)[:-1],
        indexing='xy'
    ) 

    # Computational fourier optics. Chapter 5, section 5.5.
    # Shorter propagation will not affect final results.
    
    k = 2 * np.pi / wvln
    if n == 1:
        c = 1 / (1j * wvln * z) * torch.exp(1j * k / (2 * z) * (x2 ** 2 + y2 ** 2))
    else:
        c = n / (1j * wvln * z) * torch.exp(1j * n * k / (2 * z) * (x2 ** 2 + y2 ** 2))
    
    u = c * ps ** 2 * ifftshift(fft2(fftshift(u)))

    # remove padding
    if padding:
        u = u[...,Wpad:-Wpad,Hpad:-Hpad]

    return u



""" Render sensor image by PSF convolution.
"""
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as nnF


# ================================================
# PSF convolution
# ================================================
def render_psf(img, psf, noise=None):
    """ Render an image with spatially-invariant PSF. Use the same PSF kernel for all pixels.  

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]
    
    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    # Convolution
    _, ks, ks = psf.shape
    padding = int(ks / 2)
    psf = torch.flip(psf, [1, 2])  # flip the PSF because nnF.conv2d use cross-correlation
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]
    img_pad = nnF.pad(img, (padding, padding, padding, padding), mode='reflect')
    img_render = nnF.conv2d(img_pad, psf, groups=img.shape[1], padding=0, bias=None)

    # Sensor noise
    if noise is not None:
        img_render += torch.randn_like(img_render) * noise

    return img_render


def render_psf_map(img, psf_map, grid, noise=None):
    """ Render an image with PSF map. Use the spatially-varying PSF kernels for the image.
    
        Args:
            img (torch.Tensor): [B, 3, H, W]
            psf_map (torch.Tensor): [3, grid*ks, grid*ks]
            grid (int): grid number

        Returns:
            render_img (torch.Tensor): [B, C, H, W]
    """
    if torch.is_tensor(img):
        assert len(img.shape) == 4, 'Input image should be [B, C, H, W]'
    else:
        img = torch.tensor((img/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    # Patch convolution
    Cpsf, Hpsf, Wpsf = psf_map.shape
    assert Hpsf % grid == 0 and Wpsf % grid == 0, 'PSF map size should be divisible by grid'
    ks = int(Hpsf / grid)
    assert ks % 2 == 1, 'PSF kernel size should be odd'

    B, C, H, W = img.shape
    assert C == Cpsf, 'PSF map should have the same channel as image'
    
    pad = int((ks-1)/2)
    patch_size = int(H/grid)
    img_pad = nnF.pad(img, (pad, pad, pad, pad), mode='reflect')
    
    render_img = torch.zeros_like(img)
    for i in range(grid):
        for j in range(grid):
            psf = psf_map[:, i*ks:(i+1)*ks, j*ks:(j+1)*ks]
            psf = torch.flip(psf, [1, 2]).unsqueeze(1)  # shape [C, 1, ks, ks]
            
            h_low, w_low = int(i/grid*H), int(j/grid*W)
            h_high, w_high = int((i+1)/grid*H), int((j+1)/grid*W)
            
            # Consider overlap to avoid boundary artifacts
            img_pad_patch = img_pad[:, :, h_low:h_high+2*pad, w_low:w_high+2*pad]
            render_patch = nnF.conv2d(img_pad_patch, psf, groups=img.shape[1], padding='valid', bias=None)
            render_img[:, :, h_low:h_high, w_low:w_high] = render_patch

    # Sensor noise
    if noise is not None:
        render_img += torch.randn_like(render_img) * noise

    return render_img


def local_psf_render(input, psf, kernel_size=11, noise=None):
    """ Render an image with local PSF. Use the different PSF kernel for different pixels (folding approach).
    
        Application example: Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local PSFs (1, H, W, ks, ks)
        kernel_size (int): Size of the PSFs. Defaults to 11.

    Returns:
        output (Tensor): Rendered image (N, C, H, W)
    """
    # Folding for convolution
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    B, C, H, W = input.shape
    pad = int((kernel_size - 1) / 2)

    # 1. Pad the input with replicated values
    inp_pad = nnF.pad(input, pad=(pad, pad, pad, pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = psf.reshape(-1, kernel_size, kernel_size)
    kernels_rgb = torch.stack(C * [kernels], 1)
    # 3. Unfold input
    inp_unf = nnF.unfold(inp_pad, (kernel_size, kernel_size))   
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(B, C, -1, H * W)
    x2 = kernels_rgb.view(B, H * W, C, -1).permute(0, 2, 3, 1)
    y = (x1 * x2).sum(2)
    # 5. Fold and return
    img = nnF.fold(y, (H, W), (1, 1))

    # Sensor noise
    if noise is not None:
        img += torch.randn_like(img) * noise
    
    return img 


def local_psf_render_high_res(input, psf, patch_size=[320, 480], kernel_size=11, noise=None):
    """ Patch-based rendering with local PSF. Use the different PSF kernel for different pixels.
    """
    B, C, H, W = input.shape
    img_render = torch.zeros_like(input)
    for pi in range(int(np.ceil(H/patch_size[0]))):    # int function here is not accurate
        for pj in range(int(np.ceil(W/patch_size[1]))):
            low_i = pi * patch_size[0]
            up_i = min((pi+1)*patch_size[0], H)
            low_j = pj * patch_size[1]
            up_j =  min((pj+1)*patch_size[1], W)

            img_patch = input[:, :, low_i:up_i, low_j:up_j]
            psf_patch = psf[:, low_i:up_i, low_j:up_j, :, :]

            img_render[:, :, low_i:up_i, low_j:up_j] = local_psf_render(img_patch, psf_patch, kernel_size=kernel_size, noise=noise)
    
    return img_render


# ================================================
# PSF map operations
# ================================================
def crop_psf_map(psf_map, grid, ks_crop, psf_center=None):
    """ Crop the center part of each PSF patch.

    Args:
        psf_map (torch.Tensor): [C, grid*ks, grid*ks]
        grid (int): grid number
        ks_crop (int): cropped PSF kernel size
        psf_center (torch.Tensor): (grid, grid, 2) center of the PSF patch

    Returns:
        psf_map_crop (torch.Tensor): [C, grid*ks_crop, grid*ks_crop]
    """
    if len(psf_map.shape) == 4:
        psf_map = psf_map.squeeze(0)
    C, H, W = psf_map.shape
    assert H % grid == 0 and W % grid == 0, 'PSF map size should be divisible by grid'
    ks = int(H / grid)
    assert ks % 2 == 1, 'PSF kernel size should be odd'

    psf_map_crop = torch.zeros((C, grid*ks_crop, grid*ks_crop)).to(psf_map.device)
    for i in range(grid):
        for j in range(grid):
            psf = psf_map[:, i*ks:(i+1)*ks, j*ks:(j+1)*ks]
            
            # Without re-center
            if psf_center is None:
                psf_crop = psf[:, int((ks-ks_crop)/2):int((ks+ks_crop)/2), int((ks-ks_crop)/2):int((ks+ks_crop)/2)]
            else:
                raise Exception('Not tested')
                psf_crop = psf[:, psf_center[0]-int((ks_crop-1)/2):psf_center[0]+int((ks_crop+1)/2), psf_center[1]-int((ks_crop-1)/2):psf_center[1]+int((ks_crop+1)/2)]

            # Normalize cropped PSF
            psf_crop[0, :, :] = psf_crop[0, :, :] / torch.sum(psf_crop[0, :, :])
            psf_crop[1, :, :] = psf_crop[1, :, :] / torch.sum(psf_crop[1, :, :])
            psf_crop[2, :, :] = psf_crop[2, :, :] / torch.sum(psf_crop[2, :, :])
            
            # Put cropped PSF into the map
            psf_map_crop[:, i*ks_crop:(i+1)*ks_crop, j*ks_crop:(j+1)*ks_crop] = psf_crop

    return psf_map_crop


def interp_psf_map(psf_map, grid_old, grid_new):
    """ Interpolate the PSF map from [C, grid_old*ks, grid_old*ks] to [C, grid_new*ks, grid_new*ks].
    """
    C, H, W = psf_map.shape
    assert H % grid_old == 0 and W % grid_old == 0, 'PSF map size should be divisible by grid'
    ks = int(H / grid_old)
    assert ks % 2 == 1, 'PSF kernel size should be odd'

    # Reshape from [C, grid*ks, grid*ks] to [grid_old, grid_old, C, ks, ks]
    psf_map_interp = psf_map.reshape(C, grid_old, ks, grid_old, ks).permute(1, 3, 0, 2, 4)#.reshape(grid_old, grid_old, C, ks, ks)

    # Reshape from [grid_old, grid_old, C, ks, ks] to [ks*ks, C, grid_old, grid_old]
    psf_map_interp = psf_map_interp.permute(3, 4, 2, 0, 1).reshape(ks*ks, C, grid_old, grid_old)

    # Interpolate from [ks*ks, C, grid_old, grid_old] to [ks*ks, C, grid_new, grid_new]
    psf_map_interp = nnF.interpolate(psf_map_interp, size=(grid_new, grid_new), mode='bilinear', align_corners=True)

    # Reshape from [ks*ks, C, grid_new, grid_new] to [C, grid_new*ks, grid_new*ks]
    psf_map_interp = psf_map_interp.reshape(ks, ks, C, grid_new, grid_new).permute(2, 3, 0, 4, 1).reshape(C, grid_new*ks, grid_new*ks)

    return psf_map_interp


def read_psf_map(filename, grid=10):
    """ Read PSF map from a PSF map image.

    Args:
        filename (str): path to the PSF map image
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    psf_map = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
    psf_map = torch.tensor(psf_map).permute(2, 0, 1).float() / 255.
    psf_ks = psf_map.shape[-1] // grid
    for i in range(grid):
        for j in range(grid):
            psf_map[0, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks] /= torch.sum(psf_map[0, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks])
            psf_map[1, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks] /= torch.sum(psf_map[1, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks])
            psf_map[2, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks] /= torch.sum(psf_map[2, i*psf_ks:(i+1)*psf_ks, j*psf_ks:(j+1)*psf_ks])

    return psf_map



# ================================================
# Inverse PSF calculation
# ================================================
def solve_psf(img_org, img_render, ks=51, eps=1e-6):
    """ Solve PSF, where img_render = img_org * psf.

    Args:
        img_org (torch.Tensor): The object image tensor of shape [1, 3, H, W].
        img_render (torch.Tensor): The simulated/observed image tensor of shape [1, 3, H, W].
        eps (float): A small epsilon value to prevent division by zero in frequency domain.

    Returns:
        psf (torch.Tensor): The PSF tensor of shape [3, ks, ks].
    """
    # Move to frequency domain
    F_org = torch.fft.fftn(img_org, dim=[2, 3])
    F_render = torch.fft.fftn(img_render, dim=[2, 3])
    
    # Solve for F_psf in frequency domain
    F_psf = F_render / (F_org + eps)
    
    # Inverse FFT to get PSF in spatial domain
    # Here, we take the real part assuming the PSF should be real-valued
    psf = torch.fft.ifftn(F_psf, dim=[2, 3]).real
    psf = torch.fft.fftshift(psf, dim=[2, 3])

    # Crop to get PSF size [3, 51, 51]
    _, _, H, W = psf.shape
    start_h = (H - ks) // 2
    start_w = (W - ks) // 2
    psf = psf[0, :, start_h:start_h+ks, start_w:start_w+ks]

    # Normalize PSF to sum to 1
    psf = psf / torch.sum(psf, dim=[1, 2], keepdim=True)

    return psf


def solve_psf_map(img_org, img_render, ks=51, grid=10):
    """ Solve PSF map by inverse convolution.

    Args:
        img_org (torch.Tensor): [B, 3, H, W]
        img_render (torch.Tensor): [B, 3, H, W]
        ks (int): PSF kernel size
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    assert (img_org.shape[-1] == img_org.shape[-2]), 'Image should be square'
    assert (img_org.shape[-1] % grid == 0) and (img_org.shape[-2] % grid == 0), 'Image size should be divisible by grid'
    patch_size = int(img_org.shape[-1] / grid)
    psf_map = torch.zeros((3, grid*ks, grid*ks)).to(img_org.device)
    
    for i in range(grid):
        for j in range(grid):
            img_org_patch = img_org[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            img_render_patch = img_render[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            psf_patch = solve_psf(img_org_patch, img_render_patch, ks=ks)
            
            psf_map[:, i*ks:(i+1)*ks, j*ks:(j+1)*ks] = psf_patch

    return psf_map
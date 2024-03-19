"""
Render sensor image by PSF convolution.
"""
import numpy as np
import torch
import torch.nn.functional as nnF


# ================================================
# PSF convolution
# ================================================
def render_psf(img, psf):
    """ Render an image with PSF. Use the same PSF kernel for all pixels.  

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]
    
    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    _, ks, ks = psf.shape
    padding = int(ks / 2)
    psf = torch.flip(psf, [1, 2])  # flip the PSF because nnF.conv2d use cross-correlation
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]
    img_pad = nnF.pad(img, (padding, padding, padding, padding), mode='reflect')
    img_render = nnF.conv2d(img_pad, psf, groups=img.shape[1], padding=0, bias=None)
    return img_render


def render_psf_map(img, psf_map, grid):
    """ Render an image with PSF map. Use the different PSF kernel for different patches.

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

    return render_img


def local_psf_render(input, psf, kernel_size=11):
    """ Render an image with local PSF. Use the different PSF kernel for different pixels.
    
        Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local PSFs (1, H, W, ks, ks)
        kernel_size (int): Size of the PSFs. Defaults to 11.

    Returns:
        output (Tensor): Rendered image (N, C, H, W)
    """
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    b,c,h,w = input.shape
    pad = int((kernel_size-1)/2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = psf.reshape(-1, kernel_size, kernel_size)
    kernels_rgb = torch.stack(c*[kernels], 1)
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))   
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b,c,-1,h*w)
    x2 = kernels_rgb.view(b, h*w, c, -1).permute(0, 2, 3, 1)
    y = (x1*x2).sum(2)
    # 5. Fold and return
    return torch.nn.functional.fold(y,(h,w),(1,1))


def local_psf_render_high_res(input, psf, patch_size=[320, 480], kernel_size=11):
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

            img_render[:, :, low_i:up_i, low_j:up_j] = local_psf_render(img_patch, psf_patch, kernel_size=kernel_size)
    
    return img_render


# ================================================
# PSF operations
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

    # Reshape to [grid_old, grid_old, C, ks, ks]
    psf_map_interp = psf_map.reshape(C, grid_old, ks, grid_old, ks).permute(1, 3, 0, 2, 4).reshape(grid_old, grid_old, C, ks, ks)

    # Reshape to [ks*ks, C, grid_old, grid_old]
    psf_map_interp = psf_map_interp.permute(3, 4, 2, 0, 1).reshape(ks*ks, C, grid_old, grid_old)

    # Interpolate to [ks*ks, C, grid_new, grid_new]
    psf_map_interp = nnF.interpolate(psf_map_interp, size=(grid_new, grid_new), mode='bilinear', align_corners=True)

    # Reshape to [C, grid_new*ks, grid_new*ks]
    psf_map_interp = psf_map_interp.reshape(ks, ks, C, grid_new, grid_new).permute(2, 3, 0, 4, 1).reshape(C, grid_new*ks, grid_new*ks)

    return psf_map_interp

    
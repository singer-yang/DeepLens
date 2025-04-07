import torch
# import newly modified functions
from deeplens.optics.render_psf import local_psf_render,local_psf_render_high_res
import matplotlib.pyplot as plt
import torch.nn.functional as F


# old functions for comparison
def local_psf_render_old(input, psf):
    """Render an image with pixel-wise PSF. Use the different PSF kernel for different pixels (folding approach).

        Application example: Blurs image with dynamic Gaussian blur.

    Args:
        input (Tensor): The image to be blurred (B, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, Cimg, ks, ks)

    Returns:
        output (Tensor): Rendered image (B, C, H, W)
    """
    # Folding for convolution
    B, Cimg, Himg, Wimg = input.shape
    Hpsf, Wpsf, Cpsf, Ks, Ks = psf.shape
    assert Cimg == Cpsf and Himg == Hpsf and Wimg == Wpsf, (
        "Input and PSF shape mismatch"
    )
    pad = int((Ks - 1) / 2)

    # 1. Pad the input with replicated values
    inp_pad = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")

    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = psf.reshape(Himg * Wimg, Cimg, Ks, Ks)
    kernels_flip = torch.flip(kernels, [-2, -1])

    # 3. Unfold input
    inp_unf = F.unfold(inp_pad, (Ks, Ks))  # [B, C*Ks*Ks, H*W]

    # 4. Reshape for efficient computation
    inp_unf = inp_unf.view(B, Cimg, Ks * Ks, Himg * Wimg)  # [B, C, Ks*Ks, H*W]
    kernels_flip = kernels_flip.view(Himg * Wimg, Cimg, Ks * Ks)  # [H*W, 3, Ks*Ks]

    # 5. Use einsum for efficient batch-wise multiplication and summation
    # This computes the dot product between each unfolded patch and its corresponding kernel
    # for each batch and channel
    y = torch.zeros(B, Cimg, Himg * Wimg, device=input.device)

    for b in range(B):  # Still need one loop for batch, but channels are vectorized
        # einsum: 'ckp,pck->cp' means:
        # c: channel dimension
        # k: kernel elements (Ks*Ks)
        # p: pixel positions (H*W)
        # Multiply corresponding elements and sum over k
        y[b] = torch.einsum("ckp,pck->cp", inp_unf[b], kernels_flip)

    # 6. Fold and return
    img = F.fold(y, (Himg, Wimg), (1, 1))
    return img

def local_psf_render_high_res_old(input, psf, patch_num=[4, 4], overlap=0.2):
    """Render an image with pixel-wise PSF using patch-wise rendering. Overlapping windows are used to avoid boundary artifacts.

    Args:
        input (Tensor): The image to be blurred (N, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, 3, ks, ks)
        patch_num (list): Number of patches in each dimension. Defaults to [4, 4].
        overlap (float): Fraction of overlap between adjacent patches (0-1). Defaults to 0.2.

    Returns:
        Tensor: Rendered image with same shape as input.
    """
    B, Cimg, Himg, Wimg = input.shape
    Hpsf, Wpsf, Cpsf, Ks, Ks = psf.shape
    assert Cimg == Cpsf and Himg == Hpsf and Wimg == Wpsf, (
        "Input and PSF shape mismatch"
    )

    # Calculate base patch size
    base_patch_h = Himg // patch_num[0]
    base_patch_w = Wimg // patch_num[1]

    # Calculate overlap in pixels
    overlap_h = int(base_patch_h * overlap)
    overlap_w = int(base_patch_w * overlap)

    # Initialize output and weight accumulation tensors
    img_render = torch.zeros_like(input)
    weight_accumulation = torch.zeros((B, 1, Himg, Wimg), device=input.device)

    # Create weight mask for blending (higher weight in center, lower at edges)
    def create_weight_mask(h, w):
        y = torch.linspace(0, 1, h, device=input.device)
        x = torch.linspace(0, 1, w, device=input.device)

        # Create 2D weight grid (higher in center, lower at edges)
        y = torch.min(y, 1 - y) * 2  # Transform to [0->1->0]
        x = torch.min(x, 1 - x) * 2  # Transform to [0->1->0]

        # Create 2D weight grid
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        # Combine weights (multiply or min for smoother transition)
        weights = torch.min(y_grid, x_grid).unsqueeze(0).unsqueeze(0)

        # Apply non-linearity for sharper transition
        weights = weights**2

        return weights

    # Process each patch with overlap
    for pi in range(patch_num[0]):
        for pj in range(patch_num[1]):
            # Calculate patch boundaries with overlap
            low_i = max(0, pi * base_patch_h - overlap_h)
            up_i = min(Himg, (pi + 1) * base_patch_h + overlap_h)
            low_j = max(0, pj * base_patch_w - overlap_w)
            up_j = min(Wimg, (pj + 1) * base_patch_w + overlap_w)

            # Extract patches
            img_patch = input[:, :, low_i:up_i, low_j:up_j]
            psf_patch = psf[low_i:up_i, low_j:up_j, :, :, :]

            # Process patch
            rendered_patch = local_psf_render_old(img_patch, psf_patch)

            # Create weight mask for this patch
            patch_h, patch_w = up_i - low_i, up_j - low_j
            weight_mask = create_weight_mask(patch_h, patch_w)

            # Accumulate weighted result
            img_render[:, :, low_i:up_i, low_j:up_j] += rendered_patch * weight_mask
            weight_accumulation[:, :, low_i:up_i, low_j:up_j] += weight_mask

    # Normalize by accumulated weights to blend patches
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    img_render = img_render / (weight_accumulation + epsilon)

    return img_render


if __name__ == "__main__":
    H,W,C,Ks,Ks = 20,20,1,3,3
    p1x, p1y = 10, 15
    p2x, p2y = 15, 5
    
    img = torch.zeros(1,1,H,W).float() # [1,1,H,W]
    img[0,0,p1y,p1x] = 1
    img[0,0,p2y,p2x] = 1
    
    psf = torch.zeros(H,W,C,Ks,Ks).float() # [H,W,C,Ks,Ks]
    k1 =  torch.tensor([[0,0,0],[1,1,0],[0,1,0]])*3 # triangle PSF down_left
    k2 =  torch.tensor([[0,1,0],[0,1,1],[0,0,0]]) # triangle PSF up_right
    psf[p1y,p1x,0] = k1 # triangle PSF
    psf[p2y,p2x,0] = k2 # triangle PSF

    # psf[p1y-1:p1y+2,p1x-1:p1x+2,0] = k1
    # psf[p2y-1:p2y+2,p2x-1:p2x+2,0] = k2

    # render_img = local_psf_render_scatter(img,psf)
    
    render_img = local_psf_render(img, psf)
    render_img_hr = local_psf_render_high_res(img, psf, patch_num=[3, 3])

    # old implementation
    render_img_old = local_psf_render_old(img, psf)
    render_img_hr_old = local_psf_render_high_res_old(img, psf, patch_num=[4, 4])

    img_scatter = torch.zeros(H+2,W+2).float() # [H,W]
    img_scatter[p1y:p1y+3,p1x:p1x+3] += k1
    img_scatter[p2y:p2y+3,p2x:p2x+3] += k2
    img_scatter = img_scatter[1:-1, 1:-1]

    # visualize the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img[0,0].cpu().numpy(), cmap="gray")
    plt.title("Input Image")
    # plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(img_scatter.cpu().numpy(), cmap="gray")
    plt.title("Scatter PSF (GT)")
    # plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(render_img[0,0].cpu().numpy(), cmap="gray")
    plt.title("Rendered Image")
    # plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(render_img_hr[0,0].cpu().numpy(), cmap="gray")
    plt.title("Rendered Image High Res")
    # plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(render_img_old[0,0].cpu().numpy(), cmap="gray")
    plt.title("Rendered Image Old")
    # plt.axis("off")
    plt.subplot(2, 3, 6)
    plt.imshow(render_img_hr_old[0,0].cpu().numpy(), cmap="gray")
    plt.title("Rendered Image High Res Old")
    # plt.axis("off")
    plt.tight_layout()
    # plt.show()
    # save the figure
    plt.savefig("compare_psf_render.png", dpi=300)


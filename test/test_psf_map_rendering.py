"""
Test script to verify the accuracy of psf_map rendering method.

This test uses the EDOF dataset with:
- All-in-focus image: datasets/edof/example1_aif.png
- Depth map: datasets/edof/example1_depth.npy
- Ground truth defocus image: datasets/edof/example1_left.png

Camera setup:
- Lens: Yongnuo 50mm f/1.8
- Sensor resolution: [960, 720]
- Sensor width: 32mm
- F-number: 2.8
- Focus distance: 1.78m (1780mm)

Note:
    The psf_map method uses a grid-based approach where the image is divided into
    patches and each patch uses a different PSF. This can create visible block
    artifacts at patch boundaries. For smoother results, use finer grids or
    consider the psf_patch method (without spatial variation).
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from deeplens import GeoLens
from deeplens.utils import batch_psnr, batch_ssim


# Test configuration
LENS_PATH = "datasets/lenses/camera/yongnuo_50mm_f1.8.json"
AIF_IMAGE_PATH = "datasets/edof/example1_aif.png"
DEPTH_MAP_PATH = "datasets/edof/example1_depth.npy"
GT_DEFOCUS_PATH = "datasets/edof/example1_left.png"

# Camera parameters
SENSOR_RES = (960, 720)  # (W, H)
SENSOR_WIDTH_MM = 32.0
F_NUMBER = 1.8
FOCUS_DISTANCE_MM = -1780.0  # Negative for DeepLens convention (object space)

# Output directory
OUTPUT_DIR = "test/test_outputs/psf_map_rendering"


def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an image and convert to tensor [B, C, H, W] in linear space."""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    
    # Convert sRGB to linear space
    img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    
    # Convert to tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor.float()


def tensor_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert linear tensor to sRGB space."""
    tensor = tensor.clamp(0, 1)
    srgb = torch.where(tensor <= 0.0031308, 
                       12.92 * tensor, 
                       1.055 * tensor.pow(1 / 2.4) - 0.055)
    return srgb


def load_depth_map(path: str, device: torch.device) -> torch.Tensor:
    """Load depth map and convert to tensor [B, 1, H, W] in millimeters."""
    depth = np.load(path).astype(np.float32)
    
    # Convert from meters to millimeters
    depth_mm = depth * 1000.0
    
    # Convert to tensor [B, 1, H, W]
    depth_tensor = torch.from_numpy(depth_mm).unsqueeze(0).unsqueeze(0).to(device)
    return depth_tensor


def setup_lens(device: torch.device) -> GeoLens:
    """Setup lens with specified camera parameters."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lens_path = os.path.join(project_root, LENS_PATH)
    
    # Load lens
    lens = GeoLens(filename=lens_path, device=device)
    
    # Calculate sensor height to maintain aspect ratio
    sensor_height_mm = SENSOR_WIDTH_MM * SENSOR_RES[1] / SENSOR_RES[0]
    sensor_size = (SENSOR_WIDTH_MM, sensor_height_mm)
    
    # Set sensor parameters
    lens.set_sensor(sensor_size=sensor_size, sensor_res=SENSOR_RES)
    
    # Set f-number
    lens.set_fnum(F_NUMBER)
    
    # Refocus to target distance
    lens.refocus(foc_dist=FOCUS_DISTANCE_MM)
    
    lens.to(device)
    
    print("Lens setup complete:")
    print(f"  - Focal length: {lens.foclen:.2f} mm")
    print(f"  - F-number: {lens.fnum:.2f}")
    print(f"  - Sensor size: {lens.sensor_size}")
    print(f"  - Sensor resolution: {lens.sensor_res}")
    print(f"  - d_sensor: {lens.d_sensor:.2f} mm")
    
    return lens


def test_psf_map_rendering():
    """Test the accuracy of psf_map rendering method."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all-in-focus image
    aif_path = os.path.join(project_root, AIF_IMAGE_PATH)
    img_aif = load_image_as_tensor(aif_path, device)
    print(f"All-in-focus image shape: {img_aif.shape}, range: [{img_aif.min():.3f}, {img_aif.max():.3f}]")
    
    # Load depth map
    depth_path = os.path.join(project_root, DEPTH_MAP_PATH)
    depth_map = load_depth_map(depth_path, device)
    print(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min():.1f}, {depth_map.max():.1f}] mm")
    
    # Load ground truth defocus image
    gt_path = os.path.join(project_root, GT_DEFOCUS_PATH)
    img_gt = load_image_as_tensor(gt_path, device)
    print(f"Ground truth image shape: {img_gt.shape}, range: [{img_gt.min():.3f}, {img_gt.max():.3f}]")
    
    # Setup lens
    lens = setup_lens(device)
    
    # Render using psf_map method
    print("\nRendering with psf_map method...")
    img_render = lens.render_rgbd(
        img_obj=img_aif,
        depth_map=depth_map,
        method="psf_map",
        psf_grid=(10, 10),
        psf_ks=51,
        num_depth=32,
        interp_mode="disparity"
    )
    print(f"Rendered image shape: {img_render.shape}, range: [{img_render.min():.3f}, {img_render.max():.3f}]")
    
    # Calculate metrics
    psnr = batch_psnr(img_render, img_gt).item()
    ssim = batch_ssim(img_render, img_gt).item()
    
    print("\n=== Accuracy Metrics ===")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Save results
    # Convert to sRGB for saving
    img_aif_srgb = tensor_to_srgb(img_aif)
    img_render_srgb = tensor_to_srgb(img_render)
    img_gt_srgb = tensor_to_srgb(img_gt)
    
    save_image(img_aif_srgb, os.path.join(output_dir, "01_input_aif.png"))
    save_image(img_render_srgb, os.path.join(output_dir, "02_rendered_defocus.png"))
    save_image(img_gt_srgb, os.path.join(output_dir, "03_ground_truth.png"))
    
    # Calculate and save error map
    error_map = (img_render - img_gt).abs().mean(dim=1, keepdim=True)
    error_map_normalized = error_map / error_map.max() if error_map.max() > 0 else error_map
    save_image(error_map_normalized, os.path.join(output_dir, "04_error_map.png"))
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # All-in-focus
    axes[0, 0].imshow(img_aif_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("All-in-Focus Input")
    axes[0, 0].axis("off")
    
    # Rendered
    axes[0, 1].imshow(img_render_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title(f"Rendered (psf_map)\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    axes[0, 1].axis("off")
    
    # Ground truth
    axes[1, 0].imshow(img_gt_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("Ground Truth Defocus")
    axes[1, 0].axis("off")
    
    # Error map
    im = axes[1, 1].imshow(error_map.squeeze().cpu().numpy(), cmap="hot")
    axes[1, 1].set_title("Error Map (L1)")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"PSF Map Rendering Accuracy Test\n"
                 f"Lens: YongNuo 50mm, F/{F_NUMBER}, Focus: {-FOCUS_DISTANCE_MM/1000:.2f}m, "
                 f"Sensor: {SENSOR_RES[0]}x{SENSOR_RES[1]}, {SENSOR_WIDTH_MM}mm", 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save depth map visualization
    fig, ax = plt.subplots(figsize=(10, 7.5))
    depth_vis = depth_map.squeeze().cpu().numpy() / 1000  # Convert to meters for display
    im = ax.imshow(depth_vis, cmap="turbo")
    ax.set_title("Depth Map (meters)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")
    plt.savefig(os.path.join(output_dir, "06_depth_map.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    
    return psnr, ssim


def test_psf_patch_rendering():
    """Test the accuracy of psf_patch rendering method (no spatial variation, only depth)."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all-in-focus image
    aif_path = os.path.join(project_root, AIF_IMAGE_PATH)
    img_aif = load_image_as_tensor(aif_path, device)
    
    # Load depth map
    depth_path = os.path.join(project_root, DEPTH_MAP_PATH)
    depth_map = load_depth_map(depth_path, device)
    
    # Load ground truth defocus image
    gt_path = os.path.join(project_root, GT_DEFOCUS_PATH)
    img_gt = load_image_as_tensor(gt_path, device)
    
    # Setup lens
    lens = setup_lens(device)
    
    # Render using psf_patch method (no spatial variation)
    print("\nRendering with psf_patch method (no spatial variation)...")
    img_render = lens.render_rgbd(
        img_obj=img_aif,
        depth_map=depth_map,
        method="psf_patch",
        patch_center=(0.0, 0.0),
        psf_ks=51,
        num_depth=32,
        interp_mode="disparity"
    )
    print(f"Rendered image shape: {img_render.shape}, range: [{img_render.min():.3f}, {img_render.max():.3f}]")
    
    # Calculate metrics
    psnr = batch_psnr(img_render, img_gt).item()
    ssim = batch_ssim(img_render, img_gt).item()
    
    print("\n=== Accuracy Metrics (psf_patch) ===")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Save results
    img_render_srgb = tensor_to_srgb(img_render)
    save_image(img_render_srgb, os.path.join(output_dir, "07_rendered_psf_patch.png"))
    
    return psnr, ssim


def test_psf_map_rendering_different_grids():
    """Test the effect of different PSF grid sizes on rendering accuracy."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    aif_path = os.path.join(project_root, AIF_IMAGE_PATH)
    img_aif = load_image_as_tensor(aif_path, device)
    
    depth_path = os.path.join(project_root, DEPTH_MAP_PATH)
    depth_map = load_depth_map(depth_path, device)
    
    gt_path = os.path.join(project_root, GT_DEFOCUS_PATH)
    img_gt = load_image_as_tensor(gt_path, device)
    
    # Setup lens
    lens = setup_lens(device)
    
    # Test different grid sizes
    grid_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]
    results = []
    
    print("\n=== Testing different PSF grid sizes ===")
    for grid in grid_sizes:
        print(f"\nGrid size: {grid}")
        img_render = lens.render_rgbd(
            img_obj=img_aif,
            depth_map=depth_map,
            method="psf_map",
            psf_grid=grid,
            psf_ks=51,
            num_depth=32,
            interp_mode="disparity"
        )
        
        psnr = batch_psnr(img_render, img_gt).item()
        ssim = batch_ssim(img_render, img_gt).item()
        results.append((grid, psnr, ssim))
        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # Summary table
    print("\n=== Summary ===")
    print(f"{'Grid Size':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 40)
    for grid, psnr, ssim in results:
        print(f"{str(grid):<15} {psnr:<12.2f} {ssim:<10.4f}")
    
    return results


def compare_methods():
    """Compare psf_map and psf_patch rendering methods."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    aif_path = os.path.join(project_root, AIF_IMAGE_PATH)
    img_aif = load_image_as_tensor(aif_path, device)
    
    depth_path = os.path.join(project_root, DEPTH_MAP_PATH)
    depth_map = load_depth_map(depth_path, device)
    
    gt_path = os.path.join(project_root, GT_DEFOCUS_PATH)
    img_gt = load_image_as_tensor(gt_path, device)
    
    # Setup lens
    lens = setup_lens(device)
    
    # Render with psf_patch (no spatial variation)
    print("\nRendering with psf_patch method...")
    img_patch = lens.render_rgbd(
        img_obj=img_aif,
        depth_map=depth_map,
        method="psf_patch",
        patch_center=(0.0, 0.0),
        psf_ks=64,
        num_depth=32,
        interp_mode="disparity"
    )
    
    # Render with psf_map
    print("Rendering with psf_map method...")
    img_map = lens.render_rgbd(
        img_obj=img_aif,
        depth_map=depth_map,
        method="psf_map",
        psf_grid=(4, 4),
        psf_ks=64,
        num_depth=32,
        interp_mode="disparity"
    )
    
    # Calculate metrics
    psnr_patch = batch_psnr(img_patch, img_gt).item()
    ssim_patch = batch_ssim(img_patch, img_gt).item()
    psnr_map = batch_psnr(img_map, img_gt).item()
    ssim_map = batch_ssim(img_map, img_gt).item()
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    img_patch_srgb = tensor_to_srgb(img_patch)
    img_map_srgb = tensor_to_srgb(img_map)
    img_gt_srgb = tensor_to_srgb(img_gt)
    img_aif_srgb = tensor_to_srgb(img_aif)
    
    # All-in-focus
    axes[0, 0].imshow(img_aif_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("All-in-Focus Input")
    axes[0, 0].axis("off")
    
    # psf_patch
    axes[0, 1].imshow(img_patch_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title(f"psf_patch (no spatial variation)\nPSNR: {psnr_patch:.2f} dB, SSIM: {ssim_patch:.4f}")
    axes[0, 1].axis("off")
    
    # Ground truth
    axes[1, 0].imshow(img_gt_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")
    
    # psf_map
    axes[1, 1].imshow(img_map_srgb.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title(f"psf_map (10x10 grid)\nPSNR: {psnr_map:.2f} dB, SSIM: {ssim_map:.4f}")
    axes[1, 1].axis("off")
    
    plt.suptitle(f"Method Comparison\n"
                 f"Lens: YongNuo 50mm, F/{F_NUMBER}, Focus: {-FOCUS_DISTANCE_MM/1000:.2f}m", 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_method_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print("\n=== Method Comparison ===")
    print(f"{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 45)
    print(f"{'psf_patch':<20} {psnr_patch:<12.2f} {ssim_patch:<10.4f}")
    print(f"{'psf_map (10x10)':<20} {psnr_map:<12.2f} {ssim_map:<10.4f}")
    
    # Save individual images
    save_image(img_patch_srgb, os.path.join(output_dir, "07_rendered_psf_patch.png"))
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    # Run main test
    psnr, ssim = test_psf_map_rendering()
    
    # Compare both methods
    compare_methods()
    
    # Optionally run grid size comparison
    # test_psf_map_rendering_different_grids()

"""Visualize high-order diffraction from the PSF of a hybrid refractive-diffractive lens.

Since the PSF is calculated by propagating a wave field in free space to the sensor plane,
it should contain diffraction information of multiple diffraction orders from the DOE.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich,
    "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from deeplens.hybridlens import HybridLens


def main():
    # ==========================================================
    # Load a hybrid refractive-diffractive lens (similar to 6_hybridlens_design.py)
    # ==========================================================
    lens = HybridLens(
        filename="./datasets/lenses/hybridlens/a489_grating.json",
        dtype=torch.float64
    )
    # lens.refocus(foc_dist=-1000.0)

    # ==========================================================
    # Calculate PSF at the specified point
    # Use a larger kernel size to capture higher diffraction orders
    # ==========================================================
    point = [0.0, 0.0, -10000.0]
    ks = 1024  # Larger kernel size to capture more diffraction orders
    wvln = 0.489  # Blue wavelength in micrometers

    print(f"Calculating PSF at point {point} with kernel size {ks}...")
    psf = lens.psf(points=point, ks=ks, wvln=wvln)

    # Save the PSF image
    save_image(psf.detach().clone(), "./high_order_diff_psf.png", normalize=True)
    print("Saved PSF image to ./high_order_diff_psf.png")

    # ==========================================================
    # Plot PSF values along the Y direction (center column)
    # The grating shifts the PSF in -y direction, so we plot along y
    # ==========================================================
    center_x = psf.shape[-1] // 2

    # Extract center column profile (along y-direction)
    if psf.dim() == 3:  # [C, H, W]
        psf_center = psf[:, :, center_x].mean(dim=0).detach().cpu().numpy()
    else:  # [H, W]
        psf_center = psf[:, center_x].detach().cpu().numpy()

    # Create the plot with linear and log scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # y-axis in pixels
    y_pixels = range(len(psf_center))

    # Linear scale plot
    axes[0].plot(y_pixels, psf_center, label="HybridLens PSF", color="blue", alpha=0.8)
    axes[0].set_xlabel("Y Pixel Position")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("PSF Center Column Profile (Linear Scale) - Along Y Direction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=len(psf_center)//2, color='red', linestyle='--', alpha=0.5, label='Center')

    # Log scale plot to better visualize high-order diffraction peaks
    axes[1].semilogy(y_pixels, psf_center + 1e-10, label="HybridLens PSF", color="blue", alpha=0.8)
    axes[1].set_xlabel("Y Pixel Position")
    axes[1].set_ylabel("Intensity (log scale)")
    axes[1].set_title("PSF Center Column Profile (Log Scale) - High-Order Diffraction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=len(psf_center)//2, color='red', linestyle='--', alpha=0.5, label='Center')

    plt.tight_layout()
    plt.savefig("./high_order_diff_center_line.png", dpi=150)
    plt.close()
    print("Saved center column profile (Y direction) to ./high_order_diff_center_line.png")

    # ==========================================================
    # Additional plot: 2D PSF with intensity on log scale
    # to better visualize higher diffraction orders
    # ==========================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale 2D PSF
    if psf.dim() == 3:
        psf_2d = psf.mean(dim=0).detach().cpu().numpy()
    else:
        psf_2d = psf.detach().cpu().numpy()

    im0 = axes[0].imshow(psf_2d, cmap='hot')
    axes[0].set_title("PSF (Linear Scale)")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], label='Intensity')

    # Log scale 2D PSF - reveals high-order diffraction
    import numpy as np
    psf_log = np.log10(psf_2d + 1e-10)
    im1 = axes[1].imshow(psf_log, cmap='hot')
    axes[1].set_title("PSF (Log Scale) - High-Order Diffraction")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], label='log10(Intensity)')

    plt.tight_layout()
    plt.savefig("./high_order_diff_2d.png", dpi=150)
    plt.close()
    print("Saved 2D PSF visualization to ./high_order_diff_2d.png")

    print("\n=== High-Order Diffraction Analysis Complete ===")
    print("The PSF from the hybrid lens contains diffraction information from the DOE.")
    print("Multiple diffraction orders should be visible in the log-scale plots.")


if __name__ == "__main__":
    main()

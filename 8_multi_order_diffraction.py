"""Visualize multi-order diffraction from the PSF of a hybrid refractive-diffractive lens.

When using ray tracing model (e.g., ZEMAX), we can only trace one diffraction order at a time. We have to run multiple times with assigning different diffraction efficiencies to different orders to obtain the full results. While with ray-wave model, the information of all diffraction orders is contained in the wavefront, and we can calculate the PSF with the contribution of all diffraction orders.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens.hybridlens import HybridLens


def analyze_psf(psf, save_name="./psf"):
    """Analyze and visualize PSF with both 1D center line profile and 2D plots.

    Args:
        psf: PSF tensor with shape [H, W]
        save_name: Base name for saving output files (default: "./psf")
    """
    # Plot PSF values along the Y direction (center column)
    center_x = psf.shape[-1] // 2

    # Extract center column profile (along y-direction)
    psf_center = psf[:, center_x].detach().cpu().numpy()

    # Create the plot with linear and log scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # y-axis in pixels
    y_pixels = range(len(psf_center))

    # Linear scale plot
    axes[0].plot(y_pixels, psf_center, color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Y Pixel Position")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("PSF Center Profile (Linear)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        x=len(psf_center) // 2,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Center",
    )

    # Log scale plot to better visualize high-order diffraction peaks
    axes[1].semilogy(
        y_pixels,
        psf_center + 1e-10,
        color="#3498db",
        alpha=0.8,
    )
    axes[1].set_xlabel("Y Pixel Position")
    axes[1].set_ylabel("Intensity (log scale)")
    axes[1].set_title("PSF Center Profile (Log)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(
        x=len(psf_center) // 2,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Center",
    )

    plt.tight_layout()
    plt.savefig(f"{save_name}_center_line.png", dpi=150)
    plt.close()
    print(f"Saved center column profile to {save_name}_center_line.png")

    # Plot 2D PSF
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to numpy
    psf_2d = psf.detach().cpu().numpy()

    # Linear scale 2D PSF
    im0 = axes[0].imshow(psf_2d, cmap="hot")
    axes[0].set_title("PSF (Linear)")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], label="Intensity")

    # Log scale 2D PSF - reveals high-order diffraction
    psf_log = np.log10(psf_2d + 1e-10)
    im1 = axes[1].imshow(psf_log, cmap="hot")
    axes[1].set_title("PSF (Log)")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], label="log10(Intensity)")

    plt.tight_layout()
    plt.savefig(f"{save_name}_2d.png", dpi=150)
    plt.close()
    print(f"Saved 2D PSF visualization to {save_name}_2d.png")


def main():
    # Load a hybrid refractive-diffractive lens
    # The grating (DOE) is designed for 0.55um by default. So the 0.55um PSF has the highest 1sr-order diffraction efficiency.
    lens = HybridLens(
        filename="./datasets/lenses/hybridlens/a489_grating.json", dtype=torch.float64
    )

    # Calculate PSF at the specified point for multiple wavelengths
    ks = 1024
    point = [0.0, 0.0, -10000.0]
    wvln_ls = [0.48, 0.55, 0.65]
    for wvln in wvln_ls:
        print(f"Calculating PSF at point {point} for wavelength {wvln}...")
        psf = lens.psf(points=point, ks=ks, wvln=wvln)

        analyze_psf(psf, save_name=f"./psf_{wvln}")


if __name__ == "__main__":
    main()

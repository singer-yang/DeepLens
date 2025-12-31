"""Visualize multi-order diffraction from the PSF of a hybrid refractive-diffractive lens.

The PSF is calculated by propagating the wave field after DOE (grating) to the sensor plane. It contains diffraction information of multiple diffraction orders from the DOE.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens.hybridlens import HybridLens


def main():
    # Load a hybrid refractive-diffractive lens
    lens = HybridLens(
        filename="./datasets/lenses/hybridlens/a489_grating.json", dtype=torch.float64
    )

    # Calculate PSF at the specified point for multiple wavelengths
    point = [0.0, 0.0, -10000.0]
    ks = 1024
    wavelengths = [0.48, 0.55, 0.65]

    for wvln in wavelengths:
        print(f"Calculating PSF at point {point} for wavelength {wvln}...")
        psf = lens.psf(points=point, ks=ks, wvln=wvln)

        # Plot PSF values along the Y direction (center column)
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
        axes[0].plot(
            y_pixels, psf_center, label=f"WL={wvln}um", color="#3498db", alpha=0.8
        )
        axes[0].set_xlabel("Y Pixel Position")
        axes[0].set_ylabel("Intensity")
        axes[0].set_title(f"PSF Center Profile (Linear) - WL {wvln}")
        axes[0].legend()
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
            label=f"WL={wvln}um",
            color="#3498db",
            alpha=0.8,
        )
        axes[1].set_xlabel("Y Pixel Position")
        axes[1].set_ylabel("Intensity (log scale)")
        axes[1].set_title(f"PSF Center Profile (Log) - WL {wvln}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(
            x=len(psf_center) // 2,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Center",
        )

        plt.tight_layout()
        plt.savefig(f"./high_order_diff_center_line_{wvln}.png", dpi=150)
        plt.close()
        print(
            f"Saved center column profile to ./high_order_diff_center_line_{wvln}.png"
        )

        # Plot 2D PSF
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale 2D PSF
        if psf.dim() == 3:
            psf_2d = psf.mean(dim=0).detach().cpu().numpy()
        else:
            psf_2d = psf.detach().cpu().numpy()

        im0 = axes[0].imshow(psf_2d, cmap="hot")
        axes[0].set_title(f"PSF (Linear) WL {wvln}")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        plt.colorbar(im0, ax=axes[0], label="Intensity")

        # Log scale 2D PSF - reveals high-order diffraction
        psf_log = np.log10(psf_2d + 1e-10)
        im1 = axes[1].imshow(psf_log, cmap="hot")
        axes[1].set_title(f"PSF (Log) WL {wvln}")
        axes[1].set_xlabel("X (pixels)")
        axes[1].set_ylabel("Y (pixels)")
        plt.colorbar(im1, ax=axes[1], label="log10(Intensity)")

        plt.tight_layout()
        plt.savefig(f"./high_order_diff_2d_{wvln}.png", dpi=150)
        plt.close()
        print(f"Saved 2D PSF visualization to ./high_order_diff_2d_{wvln}.png")


if __name__ == "__main__":
    main()

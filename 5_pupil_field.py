"""Calculates the pupil field (wavefront) of the lens for a point object in space by coherent ray tracing.

Note: Wavefront error is the relative error between the actual wavefront and the ideal spherical wavefront. In commercial software (e.g., Zemax), wavefront error is calculated by interpolation, which requires a low-frequency wavefront aberration. While in DeepLens, we doesnot rely on interpolation and the calculation is also accurate for high-frequency wavefront.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from deeplens import GeoLens


def main():
    # Better to use a high sensor resolution (4000x4000 is roughly acceptable, but higher is better)
    lens = GeoLens(
        filename="./datasets/lenses/cellphone/cellphone68deg.json",
        dtype=torch.float64,
    )
    lens.set_sensor_res(sensor_res=(8000, 8000))

    # ==========================================================
    # Calculate the exit-pupil field by coherent ray tracing
    # ==========================================================
    wavefront, _ = lens.pupil_field(
        point=torch.tensor([0.0, 0.0, -10000.0]), spp=20_000_000
    )
    save_image(wavefront.angle(), "./wavefront_phase.png")
    save_image(torch.abs(wavefront), "./wavefront_amp.png")

    # ==========================================================
    # Compare three different PSFs:
    #   1. geometric PSF
    #   2. Huygens PSF
    #   3. Exit-pupil propagated PSF (mathematically equivalent to Huygens PSF)
    # ==========================================================
    psf_coherent = lens.psf_coherent(torch.tensor([0.0, 0.4, -10000.0]), ks=64)
    save_image(psf_coherent, "./psf_coherent.png", normalize=True)

    psf_incoherent = lens.psf(torch.tensor([0.0, 0.4, -10000.0]), ks=64)
    save_image(psf_incoherent, "./psf_incoherent.png", normalize=True)

    psf_huygens = lens.psf_huygens(torch.tensor([0.0, 0.4, -10000.0]), ks=64)
    save_image(psf_huygens, "./psf_huygens.png", normalize=True)

    # ==========================================================
    # Plot PSF values along the center line
    # ==========================================================
    center_y = psf_coherent.shape[-2] // 2

    # Extract center line profiles (sum over channels if RGB)
    if psf_coherent.dim() == 3:  # [C, H, W]
        coherent_center = psf_coherent[:, center_y, :].mean(dim=0).cpu().numpy()
        incoherent_center = psf_incoherent[:, center_y, :].mean(dim=0).cpu().numpy()
        huygens_center = psf_huygens[:, center_y, :].mean(dim=0).cpu().numpy()
    else:  # [H, W]
        coherent_center = psf_coherent[center_y, :].cpu().numpy()
        incoherent_center = psf_incoherent[center_y, :].cpu().numpy()
        huygens_center = psf_huygens[center_y, :].cpu().numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale plot
    axes[0].plot(coherent_center, label="Pupil PSF", alpha=0.8)
    axes[0].plot(incoherent_center, label="Geometric PSF", alpha=0.8)
    axes[0].plot(huygens_center, label="Huygens PSF", alpha=0.8)
    axes[0].set_xlabel("Pixel Position")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("PSF Center Line Profile (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log scale plot to better visualize peaks and diffraction orders
    axes[1].semilogy(coherent_center + 1e-10, label="Pupil PSF", alpha=0.8)
    axes[1].semilogy(incoherent_center + 1e-10, label="Geometric PSF", alpha=0.8)
    axes[1].semilogy(huygens_center + 1e-10, label="Huygens PSF", alpha=0.8)
    axes[1].set_xlabel("Pixel Position")
    axes[1].set_ylabel("Intensity (log scale)")
    axes[1].set_title("PSF Center Line Profile (Log Scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./psf_center_line_profile.png", dpi=150)
    plt.close()
    print("Saved PSF center line profile to ./psf_center_line_profile.png")


if __name__ == "__main__":
    main()

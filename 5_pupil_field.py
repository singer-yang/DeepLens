"""Calculates the pupil field (wavefront) of the lens for a point object in space by coherent ray tracing. 

Note: Wavefront error is the relative error between the actual wavefront and the ideal spherical wavefront. In commercial software (e.g., Zemax), wavefront error is calculated by interpolation, which requires a low-frequency wavefront aberration. While in DeepLens, we doesnot rely on interpolation and the calculation is also accurate for high-frequency wavefront.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import torch
from torchvision.utils import save_image

from deeplens import GeoLens


def main():
    # Better to use a high sensor resolution (4000x4000 is roughly acceptable, but higher is better)
    lens = GeoLens(filename="./datasets/lenses/cellphone/cellphone80deg.json", dtype=torch.float64)
    lens.set_sensor_res(sensor_res=(8000, 8000))

    # Calculate the pupil field
    wavefront, _ = lens.pupil_field(
        point=torch.tensor([0.0, 0.0, -10000.0]), spp=100000000
    )
    save_image(wavefront.angle(), "./wavefront_phase.png")
    save_image(torch.abs(wavefront), "./wavefront_amp.png")

    # Compare coherent and incoherent PSFs
    psf_coherent = lens.psf_coherent(torch.tensor([0.0, 0.0, -10000.0]), ks=64)
    save_image(psf_coherent, "./psf_coherent.png", normalize=True)
    psf_incoherent = lens.psf(torch.tensor([0.0, 0.0, -10000.0]), ks=64)
    save_image(psf_incoherent, "./psf_incoherent.png", normalize=True)


if __name__ == "__main__":
    main()
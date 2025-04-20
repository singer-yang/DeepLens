"""
Calculates the pupil field of a lens at a given point in space by coherent ray tracing.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import torch
from torchvision.utils import save_image

from deeplens import GeoLens


def main():
    # Better to use a high sensor resolution (4000x4000 is small!)
    lens = GeoLens(filename="./lenses/cellphone/cellphone80deg.json")
    lens.set_sensor(sensor_res=[4000, 4000], sensor_size=lens.sensor_size)
    lens.double()

    # Calculate the pupil field
    wavefront, _ = lens.pupil_field(
        point=torch.tensor([0.0, 0.0, -10000.0]), spp=10000000
    )
    save_image(wavefront.angle(), "./wavefront_phase.png")
    save_image(torch.abs(wavefront), "./wavefront_amp.png")

    # Compare coherent and incoherent PSFs
    psf_coherent = lens.psf_coherent(torch.tensor([0.0, 0.0, -10000.0]), ks=101)
    save_image(psf_coherent, "./psf_coherent.png", normalize=True)
    psf_incoherent = lens.psf(torch.tensor([0.0, 0.0, -10000.0]), ks=101)
    save_image(psf_incoherent, "./psf_incoherent.png", normalize=True)


if __name__ == "__main__":
    main()

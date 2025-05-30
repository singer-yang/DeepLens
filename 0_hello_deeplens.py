"""
"Hello, world!" for DeepLens.

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.

Technical Paper:
    [1] Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
    [2] Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A differentiable engine for Deep Lens design of computational imaging systems," IEEE TCI 2023.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

from deeplens import GeoLens


def main():
    lens = GeoLens(filename="./lenses/camera/ef35mm_f2.0.json")
    # lens = GeoLens(filename="./lenses/camera/ef35mm_f2.0.zmx")
    # lens = GeoLens(filename='./lenses/cellphone/cellphone80deg.json')
    # lens = GeoLens(filename='./lenses/zemax_double_gaussian.zmx')

    lens.analysis(render=True)

    lens.write_lens_zmx()
    lens.write_lens_json()

if __name__ == "__main__":
    main()

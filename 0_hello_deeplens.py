"""Hello, world! for DeepLens.

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.

Technical Paper:
    [1] Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
    [2] Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A differentiable engine for Deep Lens design of computational imaging systems," IEEE TCI 2023.
"""

from deeplens import GeoLens

lens = GeoLens(filename="./datasets/lenses/camera/ef35mm_f2.0.json")
# lens = GeoLens(filename="./datasets/lenses/camera/ef35mm_f2.0.zmx")
# lens = GeoLens(filename='./datasets/lenses/cellphone/cellphone80deg.json')
# lens = GeoLens(filename='./datasets/lenses/zemax_double_gaussian.zmx')

lens.analysis(render=True)

lens.write_lens_zmx()
lens.write_lens_json()

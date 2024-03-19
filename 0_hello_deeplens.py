""" 
"Hello, world!" for DeepLens. 

In this code, we will load a lens from a file. Then we will plot the lens setup and render a sample image.

Technical Paper:
Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint (2023)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
from deeplens import Lensgroup

def main():
    lens = Lensgroup(filename='./lenses/ef40mm_f2.8.json')
    # lens = Lensgroup(filename='./lenses/cellphone68deg.json')
    lens.analysis(render=True)

if __name__=='__main__':
    main()
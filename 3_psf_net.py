""" 
Implicit representation for a realistic lens (PSFs). 

In this code, we will train a neural network to represent the PSF of a lens system. Then we can fast calculate the spatially-varying, focus-dependent PSF of the lens for image simulation.

Input: (x_normalized, y_normalized, z_unnormalized, focus_distance)
Output: [ks, ks] PSF

Technical Paper:
Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.

Some notes: 
    (1) It is easy to modify the code to support RGB PSFs.
    (2) You can also choose other network architectures.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import os
from datetime import datetime
from deeplens.psfnet import PSFNet
from deeplens.utils import set_logger, set_seed

result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-psfnet'
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)
set_seed(0)

if __name__ == "__main__":

    psfnet = PSFNet(filename='./lenses/ef50mm.json', sensor_res=(640, 640), kernel_size=11)
    psfnet.analysis(save_name=f'{result_dir}/lens')
    psfnet.write_lens_json(f'{result_dir}/lens.json')

    psfnet.train_psfnet(iters=100000, bs=128, lr=1e-4, spp=4096, evaluate_every=100, result_dir=result_dir)
    psfnet.evaluate_psf(result_dir=result_dir)
    
    print('Finish PSF net fitting.')
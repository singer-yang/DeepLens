"""
Implicit representation for a realistic lens (PSFs). In this code, we will train a neural network to represent the PSF of a lens system. Then we can fast calculate the spatially-varying, focus-dependent PSF of the lens for image simulation.

Input: [x, y, z, focus_distance]
Output: [3, ks, ks] PSF

Technical Paper:
    Xinge Yang, Qiang Fu, Mohammed Elhoseiny and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import os
from datetime import datetime

from deeplens.psfnet import PSFNetLens
from deeplens.utils import set_logger, set_seed

result_dir = "./results/" + datetime.now().strftime("%m%d-%H%M%S") + "-PSFNet"
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)
set_seed(0)

if __name__ == "__main__":
    # Init PSFNet (I changed the network archietecture to mlpconv for better performance on large PSF kernels.)
    psfnet = PSFNetLens(
        lens_path="./lenses/camera/ef50mm_f1.8.json",
        model_name="mlpconv",
        sensor_res=(2000, 2000),
        kernel_size=64,
    )
    psfnet.lens.analysis(save_name=f"{result_dir}/lens")
    psfnet.lens.write_lens_json(f"{result_dir}/lens.json")

    # Train PSFNet
    # psfnet.load_net("./ckpts/psfnet/ef50mm_f1.8_1000x1000_ks128_mlpconv.pth")
    psfnet.train_psfnet(
        iters=20000,
        bs=256,
        lr=1e-3,
        spp=100000,
        evaluate_every=1000,
        result_dir=result_dir,
    )
    psfnet.evaluate_psf(result_dir=result_dir)

    print("Finish PSF net fitting.")

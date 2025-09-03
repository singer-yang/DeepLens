# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Represent the PSF of a lens with a neural network. Surrogate model can accelerate the calculation of PSF compared to ray tracing.

Technical Paper:
    Xinge Yang, Qiang Fu, Mohammed Elhoseiny and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
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
    # Init PSFNetLens
    psfnet = PSFNetLens(
        lens_path="./lenses/camera/ef50mm_f1.8.json",
        model_name="mlpconv3",
        sensor_res=(3000, 3000),
        kernel_size=128,
    )
    psfnet.lens.analysis(save_name=f"{result_dir}/lens")
    psfnet.lens.write_lens_json(f"{result_dir}/lens.json")

    # Train PSFNetLens
    psfnet.load_net("./results/0901-174251-PSFNet/PSFNet_last.pth")
    psfnet.train_psfnet(
        iters=50000,
        bs=256,
        lr=1e-4,
        evaluate_every=100,
        concentration_factor=4.0,
        result_dir=result_dir,
    )

    # Evaluate PSFNetLens
    psfnet.evaluate_psf(result_dir=result_dir)
    print("Finish PSF net fitting.")

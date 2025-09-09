# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Represent the spatiallly varying PSF of a lens with a neural network. Surrogate model can accelerate the calculation of PSF compared to ray tracing.

Technical Paper:
    Xinge Yang, Qiang Fu, Mohammed Elhoseiny and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
"""

import os
from datetime import datetime

from deeplens.psfnetlens import PSFNetLens
from deeplens.utils import set_logger

result_dir = "./results/" + datetime.now().strftime("%m%d-%H%M%S") + "-PSFNet"
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)

if __name__ == "__main__":
    # Init PSFNetLens
    # Input (B, 3): (fov, depth, foc_dist)
    # Output (B, 3, ks, ks): RGB PSF on y-axis at (fov, depth, foc_dist)
    psfnet_lens = PSFNetLens(
        in_chan=3,
        psf_chan=3,
        lens_path="./lenses/camera/ef50mm_f1.8.json",
        model_name="mlpconv",
        kernel_size=128,
        sensor_res=(3000, 3000),
    )
    psfnet_lens.lens.analysis(save_name=f"{result_dir}/lens")
    psfnet_lens.lens.write_lens_json(f"{result_dir}/lens.json")
    psfnet_lens.load_net("./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth")

    # Draw example PSF map
    psfnet_lens.refocus(-1200)
    psfnet_lens.draw_psf_map(
        save_name="./psf_map_net.png",
        grid=(11, 11),
        ks=64,
        depth=-1500,
        log_scale=False,
    )
    psfnet_lens.lens.draw_psf_map(
        save_name="./psf_map_lens.png",
        grid=(11, 11),
        ks=64,
        depth=-1500,
        log_scale=False,
    )

    # Training
    psfnet_lens.train_psfnet(
        iters=10000,
        evaluate_every=100,
        bs=128,
        lr=5e-5,
        result_dir=result_dir,
    )
    print("Finish PSF net fitting.")

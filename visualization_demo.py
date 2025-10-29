import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens import GeoLens

from deeplens.geolens_pkg.view_3d_gui import draw_lens_3d

# import pyvistaqt as pvqt
# from deeplens.geolens_pkg.view_3d import draw_lens_3d, save_lens_obj

R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./visualization")
args = parser.parse_args()
SAVE_DIR = args.save_dir

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# lens_config = os.path.relpath("./lenses/cellphone/cellphone68deg.json")
lens_config = os.path.relpath("./lenses/camera/ef50mm_f1.8.json")

lens = GeoLens(lens_config)

# (lens.draw_layout(os.path.join(SAVE_DIR, "lens_layout2d.png")),)
# plotter = pvqt.BackgroundPlotter()
# plotter = pv.Plotter(off_screen=True)

rfov = lens.rfov

lens.save_lens_obj(save_dir=SAVE_DIR, save_elements=True, save_rays=True)

draw_lens_3d(
    lens = lens,
    save_dir=SAVE_DIR,
    fovs=[0.0, rfov * 0.99 * 57.296],
    fov_phis=[45.0, 135.0, 225.0, 315.0],
)
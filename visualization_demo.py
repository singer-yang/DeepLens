import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens import GeoLens

from deeplens.geolens_pkg.view_3d_gui import draw_lens_3d
import os
os.environ["PYVISTA_OFF_SCREEN"] = "1"        # force off-screen renders
os.environ["PYVISTA_JUPYTER_BACKEND"] = "static"  # avoid widgets/CDN
# Optional: if you previously had DISPLAY set, clear it to prevent Qt/GLX tries
os.environ.pop("DISPLAY", None)

import pyvista as pv

# If Xvfb happens to be available, this helps many headless cases.
# (It's no-op if Xvfb isn't installed.)
try:
    pv.start_xvfb()   # starts a virtual X server if present
    print("xvfb started")
except Exception as e:
    print("xvfb not started:", e)

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


rfov = lens.rfov

# lens.save_lens_obj(save_dir=SAVE_DIR, save_elements=True, save_rays=True)


plotter = pv.Plotter(off_screen=True, notebook=True)
draw_lens_3d(
    plotter=plotter,
    lens=lens,
    save_dir="./visualization/",
    fovs=[0.0, rfov * 0.99 * 57.296],
    fov_phis=[45.0, 135.0, 225.0, 315.0],
    draw_rays=True,
)
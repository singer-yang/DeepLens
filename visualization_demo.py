from deeplens import GeoLens

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import pyvista as pv
# import pyvistaqt as pvqt

from deeplens.geolens_pkg.view_3d import draw_lens_3D

R = np.array([255,0,0])
G = np.array([0,255,0])
B = np.array([0,0,255])
SAVE_DIR = "./visualization"
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

lens_config = os.path.relpath(
    "./lenses/cellphone/cellphone68deg.json"
)

lens = GeoLens(lens_config)

lens.draw_layout(os.path.join(SAVE_DIR, "lens_2dlayout.png")),
# plotter = pvqt.BackgroundPlotter()
plotter = pv.Plotter(off_screen=True)

hfov = lens.hfov

draw_lens_3D(plotter,lens,
             fovs= [0., hfov*0.99 * 57.296],
             fov_phis = [45., 135., 225., 315.],
             is_show_bridge=False,
             save_dir=SAVE_DIR,)

# plotter.show()
# plotter.app.exec_()
plotter.screenshot(os.path.join(SAVE_DIR, "lens_3d_visualization.png"))
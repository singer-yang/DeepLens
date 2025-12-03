import argparse
import os

import numpy as np

from deeplens import GeoLens

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./visualization")
args = parser.parse_args()
SAVE_DIR = args.save_dir

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# lens_config = os.path.relpath("./lenses/cellphone/cellphone68deg.json")
lens_config = os.path.relpath("./datasets/lenses/camera/ef50mm_f1.8.json")

lens = GeoLens(lens_config)
rfov = lens.rfov

# =============================================================================
# Test 1: Save lens OBJ files (does NOT require PyVista)
# =============================================================================
print("=" * 60)
print("Test 1: save_lens_obj (no PyVista required)")
print("=" * 60)

lens.save_lens_obj(save_dir=SAVE_DIR, save_elements=True, save_rays=True, is_wrap=True)

print(f"OBJ files saved to {SAVE_DIR}")
print("Test 1 passed!\n")
breakpoint()

# =============================================================================
# Test 2: Draw lens 3D layout (requires PyVista - lazy loaded)
# =============================================================================
print("=" * 60)
print("Test 2: draw_lens_3d (PyVista required - lazy loaded)")
print("=" * 60)

# PyVista setup for headless rendering
os.environ["PYVISTA_OFF_SCREEN"] = "1"  # force off-screen renders
os.environ["PYVISTA_JUPYTER_BACKEND"] = "static"  # avoid widgets/CDN
# Optional: if you previously had DISPLAY set, clear it to prevent Qt/GLX tries
os.environ.pop("DISPLAY", None)

import pyvista as pv

# If Xvfb happens to be available, this helps many headless cases.
# (It's no-op if Xvfb isn't installed.)
try:
    pv.start_xvfb()  # starts a virtual X server if present
    print("xvfb started")
except Exception as e:
    print("xvfb not started:", e)

plotter = pv.Plotter(off_screen=True, notebook=True)
lens.draw_lens_3d(
    plotter=plotter,
    save_dir=SAVE_DIR,
    fovs=[0.0, rfov * 0.99 * 57.296],
    fov_phis=[45.0, 135.0, 225.0, 315.0],
    draw_rays=True,
)

print(f"3D layout saved to {SAVE_DIR}/lens_layout3d.png")
print("Test 2 passed!\n")

print("=" * 60)
print("All tests completed successfully!")
print("=" * 60)

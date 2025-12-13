"""
Automated lens design from scratch. This code uses RMS spot size for lens design, which is much faster than image-based lens design.

Technical Paper:
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math
import os
import random
import string
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from deeplens.geolens import GeoLens
from deeplens.geolens_pkg.utils import create_lens
from deeplens.basics import DEPTH, EPSILON, WAVE_RGB
from deeplens.utils import create_video_from_images, set_logger, set_seed


def config():
    """Config file for training."""
    # Config file
    with open("configs/2_auto_lens_design.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-AutoLens-RMS-" + random_string
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    if args["seed"] is None:
        seed = random.randint(0, 100000)
        args["seed"] = seed
    set_seed(args["seed"])

    # Log
    set_logger(result_dir)
    logging.info(f"EXP: {args['EXP_NAME']}")

    # Device
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # ==> Save config and original code
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)

    with open(f"{result_dir}/2_autolens_rms.py", "w") as f:
        with open("2_autolens_rms.py", "r") as code:
            f.write(code.read())

    return args


def curriculum_design(
    self: GeoLens,
    lrs=[1e-4, 1e-4, 1e-2, 1e-4],
    decay=0.01,
    iterations=5000,
    test_per_iter=100,
    optim_mat=False,
    match_mat=False,
    shape_control=True,
    result_dir="./results",
):
    """Optimize the lens by minimizing rms errors."""
    # Preparation
    depth = DEPTH
    num_ring = 8
    num_arm = 8
    spp = 2048

    aper_start = self.surfaces[self.aper_idx].r * 0.2
    aper_final = self.surfaces[self.aper_idx].r

    # Log
    if not logging.getLogger().hasHandlers():
        set_logger(result_dir)
    logging.info(
        f"lr:{lrs}, iterations:{iterations}, spp:{spp}, num_ring:{num_ring}, num_arm:{num_arm}."
    )

    # Optimizer
    optimizer = self.get_optimizer(lrs, decay=decay, optim_mat=optim_mat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # Training loop
    pbar = tqdm(
        total=iterations + 1, desc="Progress", postfix={"loss_rms": 0, "loss_reg": 0}
    )
    for i in range(iterations + 1):
        # =======================================
        # Evaluate the lens
        # =======================================
        if i % test_per_iter == 0:
            with torch.no_grad():
                # Curriculum learning: gradually increase aperture size
                progress = 0.5 * (1 + math.cos(math.pi * (1 - i / iterations)))
                aper_r = min(
                    aper_start + (aper_final - aper_start) * progress,
                    aper_final,
                )
                self.surfaces[self.aper_idx].update_r(aper_r)
                self.calc_pupil()

                # Correct lens shape and evaluate current design
                if i > 0:
                    if shape_control:
                        self.correct_shape()
                        # self.refocus()

                    if optim_mat and match_mat:
                        self.match_materials()

                # Save lens
                self.write_lens_json(f"{result_dir}/iter{i}.json")
                self.analysis(f"{result_dir}/iter{i}")

                # Sample new rays and calculate target centers
                rays_backup = []
                for wv in WAVE_RGB:
                    ray = self.sample_ring_arm_rays(
                        num_ring=num_ring,
                        num_arm=num_arm,
                        depth=depth,
                        spp=spp,
                        wvln=wv,
                        scale_pupil=1.10,
                    )
                    rays_backup.append(ray)

                center_ref = -self.psf_center(points=ray.o[:, :, 0, :], method="pinhole")
                center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

        # =======================================
        # Optimize lens by minimizing rms
        # =======================================
        loss_rms = []
        for wv_idx, wv in enumerate(WAVE_RGB):
            # Ray tracing to sensor, [num_grid, num_grid, num_rays, 3]
            ray = rays_backup[wv_idx].clone()
            ray = self.trace2sensor(ray)

            # Ray error to center and valid mask
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref

            # Weight mask (non-differentiable), shape of [num_grid, num_grid]
            if wv_idx == 0:
                with torch.no_grad():
                    weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                    weight_mask /= ray_valid.sum(-1) + EPSILON
                    weight_mask /= weight_mask.mean()

                    # Drop out (20% of weight mask)
                    dropout_mask = torch.rand_like(weight_mask) < 0.2
                    weight_mask = weight_mask * (~dropout_mask)

            # Loss on rms error, shape of [num_grid, num_grid]
            l_rms = (((ray_err**2).sum(-1) + EPSILON).sqrt() * ray_valid).sum(-1)
            l_rms /= ray_valid.sum(-1) + EPSILON

            # Weighted loss
            l_rms_weighted = (l_rms * weight_mask).sum()
            l_rms_weighted /= weight_mask.sum() + EPSILON
            loss_rms.append(l_rms_weighted)

        # RMS loss for all wavelengths
        loss_rms = sum(loss_rms) / len(loss_rms)

        # Add lens design constraint
        loss_reg, loss_dict = self.loss_reg()
        w_reg = 0.05
        L_total = loss_rms + w_reg * loss_reg

        # Gradient-based optimization
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]

    # Bind function
    GeoLens.curriculum_design = curriculum_design

    # Create a lens
    lens = create_lens(
        foclen=args["foclen"],
        fov=args["fov"],
        fnum=args["fnum"],
        flange=args["flange"],
        thickness=args["thickness"],
        surf_list=args["surf_list"],
        save_dir=result_dir,
    )
    lens.set_target_fov_fnum(
        rfov=args["fov"] / 2 / 57.3,
        fnum=args["fnum"],
    )
    logging.info(
        f"==> Design target: focal length {round(args['foclen'], 2)}, diagonal FoV {args['fov']}deg, F/{args['fnum']}"
    )

    # Curriculum learning with RMS errors
    # Curriculum learning is used to find an optimization path when starting from scratch, where the optimization difficulty is high and the gradients are unstable. 3000 iterations is a good starting value, while increasing the number of iterations will improve the optical performance. Also, we can choose to optimize materials in this stage.
    lens.curriculum_design(
        lrs=[float(lr) for lr in args["lrs"]],
        decay=float(args["decay"]),
        iterations=2000,
        test_per_iter=50,
        optim_mat=True,
        match_mat=False,
        shape_control=True,
        result_dir=args["result_dir"],
    )

    # Match materials and set fnum
    lens.match_materials()
    lens.set_fnum(args["fnum"])
    lens.write_lens_json(f"{result_dir}/curriculum_final.json")

    # To obtain optimal optical performance, we typically need additional training iterations. This code uses strong lens design constraints with small learning rates, making optimization slow but steadily improving optical performance. For demonstration purposes, here we only train for 3000 steps.
    lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
    lens.optimize(
        lrs=[float(lr) * 0.1 for lr in args["lrs"]],
        decay=float(args["decay"]),
        iterations=5000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        result_dir=f"{args['result_dir']}/fine-tune",
    )

    # Analyze final result
    lens.prune_surf(expand_factor=0.05)
    lens.post_computation()

    logging.info(
        f"Actual: diagonal FOV {lens.rfov}, r sensor {lens.r_sensor}, F/{lens.fnum}."
    )
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    # Create video
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)

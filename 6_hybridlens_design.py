"""
Jointly optimize refractive-diffractive lens with a differentiable ray-wave model. This code can be easily extended to end-to-end refractive-diffractive lens and network design. 

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import logging
import os
import random
import string
from datetime import datetime

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens.hybridlens import HybridLens
from deeplens.optics.loss import PSFLoss
from deeplens.utils import set_logger, set_seed


def config():
    # ==> Config
    args = {"seed": 0, "DEBUG": True}

    # ==> Result folder
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    result_dir = (
        "./results/"
        + datetime.now().strftime("%m%d-%H%M%S")
        + "-HybridLens"
        + "-"
        + random_string
    )
    args["result_dir"] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result folder: {result_dir}")

    if args["seed"] is None:
        seed = random.randint(0, 100)
        args["seed"] = seed
    set_seed(args["seed"])

    # ==> Log
    set_logger(result_dir)
    if not args["DEBUG"]:
        raise Exception("Add your wandb logging config here.")

    # ==> Device
    num_gpus = torch.cuda.device_count()
    args["num_gpus"] = num_gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args["device"] = device
    logging.info(f"Using {num_gpus} {torch.cuda.get_device_name(0)} GPU(s)")

    # ==> Save config
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)

    with open(f"{result_dir}/6_hybridlens_design.py", "w") as f:
        with open("6_hybridlens_design.py", "r") as code:
            f.write(code.read())

    return args


def main(args):
    # Create a hybrid refractive-diffractive lens
    lens = HybridLens(lens_path="./lenses/hybridlens/a489_doe.json")
    lens.double()

    # PSF optimization loop to focus blue light
    optimizer = lens.get_optimizer(doe_lr=0.1, lens_lr=[1e-4, 1e-4, 1e-1, 1e-5])
    loss_fn = PSFLoss()
    for i in tqdm(range(100 + 1)):
        psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=101, wvln=0.489)

        optimizer.zero_grad()
        loss = loss_fn(psf)
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            lens.write_lens_json(f"{args['result_dir']}/lens_iter{i}.json")
            lens.analysis(save_name=f"{args['result_dir']}/lens_iter{i}.png")
            save_image(
                psf.detach().clone(),
                f"{args['result_dir']}/psf_iter{i}.png",
                normalize=True,
            )


if __name__ == "__main__":
    args = config()
    main(args)

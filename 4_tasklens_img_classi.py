# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Task-driven lens design for image classification.

We design a lens with from scratch with only image-classification loss. This makes sure no classical lens design objective (spot size, PSF...) is used in the task-driven lens design. By doing this, we can explore "unseen" lens design space to find a lens that is optimal for a specific task, because we totally get rid of classical lens design!

Technical Paper:
    Xinge Yang, Yunfeng Nie, Fu Qiang and Wolfgang Heidrich, "Image Quality Is Not All You Want: Task-Driven Lens Design for Image Classification" Arxiv preprint 2023.
"""

import logging
import os
import random
import string
from datetime import datetime

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from deeplens import GeoLens
from deeplens.optics.psf import conv_psf
from deeplens.utils import set_logger, set_seed


def config():
    # ==> Config
    with open("configs/4_tasklens.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # ==> Result folder
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    result_dir = (
        "./results/"
        + datetime.now().strftime("%m%d-%H%M%S")
        + "-TaskLens"
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
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # ==> Save config
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)

    with open(f"{result_dir}/4_tasklens_img_classi.py", "w") as f:
        with open("4_tasklens_img_classi.py", "r") as code:
            f.write(code.read())

    return args


def get_dataset(args):
    dataset = args["train"]["dataset"]
    img_res = args["train"]["img_res"]
    bs = args["train"]["bs"]

    # ==> Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize(img_res),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(img_res),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # ==> Datset
    if dataset == "imagenet":
        train_dataset = ImageFolder(
            root=args["imagenet_train_dir"], transform=train_transform
        )
        val_dataset = ImageFolder(
            root=args["imagenet_val_dir"], transform=val_transform
        )
    elif dataset == "imagenet_local":
        train_dataset = ImageFolder(
            root=args["imagenet_train_dir_local"], transform=train_transform
        )
        val_dataset = ImageFolder(
            root=args["imagenet_val_dir_local"], transform=val_transform
        )
    else:
        raise NotImplementedError

    # ==> Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

    return train_loader, val_loader


def get_network(args):
    if args["network"]["model"] == "resnet50":
        net = timm.create_model("resnet50", pretrained=True, num_classes=1000)
    elif args["network"]["model"] == "swin_transformer":
        net = timm.create_model(
            "swin_base_patch4_window7_224_in22k", pretrained=True, num_classes=1000
        )
    elif args["network"]["model"] == "mobilenet":
        net = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, num_classes=1000
        )
    elif args["network"]["model"] == "vit":
        net = timm.create_model(
            "vit_large_patch16_224_in21k", pretrained=True, num_classes=1000
        )
    else:
        raise NotImplementedError

    # Parallel
    net = nn.DataParallel(net, device_ids=range(args["num_gpus"]))
    return net


@torch.no_grad()
def validate(lens, net, epoch, args, val_loader):
    """Test image classification accuracy."""
    # Parameters
    device = args["device"]
    result_dir = args["result_dir"]
    depth = args["train"]["depth"]
    bs = args["train"]["bs"]
    ks = args["train"]["psf_ks"]
    psf_grid = args["train"]["psf_grid"]
    points = lens.point_source_grid(
        depth=depth, grid=psf_grid * 2 - 1, quater=True
    ).reshape(-1, 3)

    # Scores
    correct = 0.0
    total = 0.0

    # Calculate PSFs
    psf = lens.psf_rgb(points=points, ks=ks, spp=4096)

    # Loop over the validation set in batches
    for _, (img_org, labels) in tqdm(enumerate(val_loader)):
        if img_org.shape[0] != bs:
            continue

        # Get images and labels
        img_org = img_org.to(device)
        labels = labels.to(device)

        # Render image with PSF map
        img_render = conv_psf(img_org, psf)
        img_render = torch.cat(img_render)
        labels = labels.repeat(psf_grid**2)

        # Forward pass and prediction
        outputs = net(img_render)
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy statistics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print validation accuracy
    acc = correct / total
    if acc > args["val_acc"]:
        args["val_acc"] = acc
        logging.info(f"Best epoch is {epoch}, best Val acc is {acc}.")
        torch.save(net.state_dict(), f"{result_dir}/classi_model_best.pth")

    logging.info("Validation Accuracy: {:.2f}%".format(100 * acc))
    if not args["DEBUG"]:
        wandb.log({"classi_acc": acc})


def train(args, lens: GeoLens, net):
    device = args["device"]
    result_dir = args["result_dir"]
    bs = args["train"]["bs"]
    ks = args["train"]["psf_ks"]
    psf_grid = args["train"]["psf_grid"]
    spp = args["train"]["spp"]
    depth = args["train"]["depth"]
    lens_lrs = [float(i) for i in args["lens"]["lr"]]
    args["val_acc"] = 0

    # ==> Dataset
    train_loader, val_loader = get_dataset(args)
    batchs = len(train_loader)
    epochs = args["train"]["epochs"]

    # ==> Optimizer and scheduler
    lens_optim = lens.get_optimizer(lrs=lens_lrs)
    lens_sche = get_cosine_schedule_with_warmup(
        lens_optim, num_warmup_steps=500, num_training_steps=batchs * epochs
    )
    # # Uncomment for End-to-End lens-network co-design
    # net_optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    # net_sche = get_cosine_schedule_with_warmup(net_optim, num_warmup_steps=500, num_training_steps=batchs*epochs)

    # ==> Loss
    cri_classi = nn.CrossEntropyLoss()

    # ==> Training
    logging.info("==> Start training.")
    points = lens.point_source_grid(depth=depth, grid=psf_grid, quater=True).reshape(
        -1, 3
    )
    for epoch in range(args["train"]["epochs"] + 1):
        # =============================
        # Evaluation
        # =============================
        if epoch % 1 == 0 and epoch > 0:
            net.eval()
            lens.correct_shape()
            lens.write_lens_json(f"{result_dir}/epoch{epoch}.json")
            lens.analysis(f"{result_dir}/epoch{epoch}")
            validate(lens, net, epoch, args, val_loader)

        # =============================
        # Training
        # =============================
        net.train()

        # ==> Task-driven lens design: a well-trained network serves as lens design objective
        for ii, (img_org, labels) in tqdm(enumerate(train_loader)):
            # Continue is wrong batch size
            if img_org.shape[0] != bs:
                continue

            # Get images and labels
            img_org = img_org.to(device)
            labels = labels.to(device)

            # Option 1: Render image with PSF map
            psf = lens.psf_rgb(
                points=points, ks=ks, center=False, spp=spp
            )  # [N, 3, ks, ks]
            img_render = []
            for psf_idx in range(psf.shape[0]):
                img_render.append(conv_psf(img_org, psf[psf_idx, ...]))
            img_render = torch.cat(img_render)  # [N * B, 3, sensor_res, sensor_res]
            labels = labels.repeat(psf.shape[0])

            # Option 2: Render image with ray tracing
            # img_render = lens.render(img_org)

            # Image classification
            labels_pred = net(img_render)

            # Loss
            L_classi = cri_classi(labels_pred, labels)
            L_reg = lens.loss_self_intersec()

            L = L_classi + 0.02 * L_reg

            # Update
            lens_optim.zero_grad()
            # net_optim.zero_grad()
            L.backward()
            lens_optim.step()
            # net_optim.step()
            lens_sche.step()
            # net_sche.step()

            if not args["DEBUG"]:
                wandb.log({"loss_class": L_classi.detach().item()})

            # Print statistics every 1000 batches
            if ii % 100 == 0 and ii > 0:
                logging.info(
                    "Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        args["train"]["epochs"],
                        ii,
                        len(train_loader),
                        L.item(),
                    )
                )
                lens.correct_shape()
                lens.write_lens_json(f"{result_dir}/epoch{epoch}_batch{ii}.json")
                lens.analysis(f"{result_dir}/epoch{epoch}_batch{ii}")

        logging.info(f"Epoch{epoch + 1} finishs.")


if __name__ == "__main__":
    args = config()

    # Lens
    lens = GeoLens(filename=args["lens"]["path"]).to(args["device"])
    lens.set_target_fov_fnum(
        hfov=args["lens"]["target_hfov"], fnum=args["lens"]["target_fnum"]
    )
    lens.write_lens_json(f"{args['result_dir']}/epoch0.json")
    lens.analysis(f"{args['result_dir']}/epoch0", render=False)

    # Network
    net = get_network(args)
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(args["device"])

    # End-to-end lens-network co-design
    train(args, lens, net)

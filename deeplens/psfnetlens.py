# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Surrogate lens model that represents the Point Spread Function (PSF) of a lens using a neural network. This surrogate model can significantly accelerate PSF calculations compared to traditional ray tracing methods.

Technical Paper:
    Xinge Yang, Qiang Fu, Mohamed Elhoseiny, and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deeplens.geolens import GeoLens
from deeplens.lens import Lens
from deeplens.network.surrogate import MLP
from deeplens.network.surrogate.psfnet_mplconv import PSFNet_MLPConv
from deeplens.basics import DEPTH, PSF_KS
from deeplens.optics.psf import conv_psf_pixel, conv_psf_pixel_high_res, rotate_psf


class PSFNetLens(Lens):
    def __init__(
        self,
        lens_path,
        in_chan=3,
        psf_chan=3,
        model_name="mlp_conv",
        kernel_size=64,
    ):
        """Initialize a PSF network lens.

        In the default settings, the PSF network takes (fov, depth, foc_dist) as input and outputs RGB PSF on y-axis at (fov, depth, foc_dist).

        Args:
            lens_path (str): Path to the lens file.
            in_chan (int): Number of input channels.
            psf_chan (int): Number of output channels.
            model_name (str): Name of the model.
            kernel_size (int): Kernel size.
        """
        super().__init__()

        # Load lens (sensor_size and sensor_res are read from the lens file)
        self.lens_path = lens_path
        self.lens = GeoLens(filename=lens_path, device=self.device)
        self.rfov = self.lens.rfov

        # Init PSF network
        self.in_chan = in_chan
        self.psf_chan = psf_chan
        self.kernel_size = kernel_size
        self.pixel_size = self.lens.pixel_size

        self.psfnet = self.init_net(
            in_chan=in_chan,
            psf_chan=psf_chan,
            kernel_size=kernel_size,
            model_name=model_name,
        )
        self.psfnet.to(self.device)

        # Object depth range
        self.d_close = -200
        self.d_far = -20000

        # Focus distance range
        # There is a minimum focal distance for each lens. For example, the Canon EF 50mm lens can only focus to 0.5m and further.
        self.foc_d_close = -500
        self.foc_d_far = -20000
        self.refocus(foc_dist=-20000)

    def set_sensor_res(self, sensor_res):
        """Set sensor resolution for both PSFNetLens and the embedded GeoLens."""
        self.lens.set_sensor_res(sensor_res)
        self.pixel_size = self.lens.pixel_size

    # ==================================================
    # Training functions
    # ==================================================
    def init_net(self, in_chan=2, psf_chan=3, kernel_size=64, model_name="mlpconv"):
        """Initialize a PSF network.

        PSF network:
            Input: [B, 3], (fov, depth, foc_dist). fov from [0, pi/2], depth from [-20000, -100], foc_dist from [-20000, -500]
            Output: psf kernel [B, 3, ks, ks]

        Args:
            in_chan (int): number of input channels
            psf_chan (int): number of output channels
            kernel_size (int): kernel size
            model_name (str): name of the network architecture

        Returns:
            psfnet (nn.Module): network
        """
        if model_name == "mlp":
            psfnet = MLP(
                in_features=in_chan,
                out_features=psf_chan * kernel_size**2,
                hidden_features=256,
                hidden_layers=8,
            )
        elif model_name == "mlpconv":
            psfnet = PSFNet_MLPConv(
                in_chan=in_chan, kernel_size=kernel_size, out_chan=psf_chan
            )
        else:
            raise Exception(f"Unsupported PSF network architecture: {model_name}.")

        return psfnet

    def load_net(self, net_path):
        """Load pretrained network.

        Args:
            net_path (str): path to load the network
        """
        # Check the correct model is loaded
        psfnet_dict = torch.load(net_path, map_location="cpu", weights_only=False)
        print(
            f"Pretrained model lens pixel size: {psfnet_dict['pixel_size']*1000.0:.1f} um, "
            f"Current lens pixel size: {self.pixel_size*1000.0:.1f} um"
        )
        print(
            f"Pretrained model lens path: {psfnet_dict['lens_path']}, "
            f"Current lens path: {self.lens_path}"
        )

        # Load the model weights
        self.psfnet.load_state_dict(psfnet_dict["psfnet_model_weights"])

    def save_psfnet(self, psfnet_path):
        """Save the PSF network.

        Args:
            psfnet_path (str): path to save the PSF network
        """
        psfnet_dict = {
            "model_name": self.psfnet.__class__.__name__,
            "in_chan": self.in_chan,
            "pixel_size": self.pixel_size,
            "kernel_size": self.kernel_size,
            "psf_chan": self.psf_chan,
            "lens_path": self.lens_path,
            "psfnet_model_weights": self.psfnet.state_dict(),
        }
        torch.save(psfnet_dict, psfnet_path)

    def train_psfnet(
        self,
        iters=100000,
        bs=128,
        lr=5e-5,
        evaluate_every=500,
        spp=16384,
        concentration_factor=2.0,
        result_dir="./results/psfnet",
    ):
        """Train the PSF surrogate network.

        Args:
            iters (int): number of training iterations
            bs (int): batch size
            lr (float): learning rate
            evaluate_every (int): evaluate every how many iterations
            spp (int): number of samples per pixel
            concentration_factor (float): concentration factor for training data sampling
            result_dir (str): directory to save the results
        """
        # Init network and prepare for training
        psfnet = self.psfnet
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.AdamW(psfnet.parameters(), lr=lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(iters) // 100, num_training_steps=iters
        )

        # Train the network
        for i in tqdm(range(iters + 1)):
            # Sample training data
            sample_input, sample_psf = self.sample_training_data(
                num_points=bs, concentration_factor=concentration_factor
            )
            sample_input, sample_psf = (
                sample_input.to(self.device),
                sample_psf.to(self.device),
            )

            # Forward pass, pred_psf: [B, 3, ks, ks]
            pred_psf = psfnet(sample_input)

            # Backward propagation
            optimizer.zero_grad()
            loss = loss_fn(pred_psf, sample_psf)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Evaluate training
            if (i + 1) % evaluate_every == 0:
                # Visualize PSFs
                n_vis = 16
                fig, axs = plt.subplots(n_vis, 2, figsize=(4, n_vis * 2))
                for j in range(n_vis):
                    psf0 = sample_psf[j, ...].detach().clone().cpu()
                    axs[j, 0].imshow(psf0.permute(1, 2, 0) * 255.0)
                    axs[j, 0].axis("off")

                    psf1 = pred_psf[j, ...].detach().clone().cpu()
                    axs[j, 1].imshow(psf1.permute(1, 2, 0) * 255.0)
                    axs[j, 1].axis("off")

                axs[0, 0].set_title("GT")
                axs[0, 1].set_title("Pred")

                fig.suptitle(f"GT/Pred PSFs at iter {i + 1}")
                plt.tight_layout()
                plt.savefig(f"{result_dir}/iter{i + 1}.png", dpi=300)
                plt.close()

                # Save network
                self.save_psfnet(f"{result_dir}/PSFNet_last.pth")

    @torch.no_grad()
    def sample_training_data(self, num_points=512, concentration_factor=2.0):
        """Sample training data for PSF surrogate network.

        Args:
            num_points (int): number of training points
            concentration_factor (float): concentration factor for training data sampling

        Returns:
            sample_input (tensor): [B, 3] tensor, (fov, depth, foc_dist).
                - fov from [0, rfov] on 0y-axis, [radians]
                - depth from [d_far, d_close], [mm]
                - foc_dist from [foc_d_far, foc_d_close], [mm]
                - We use absolute fov and depth.

            sample_psf (tensor): [B, 3, ks, ks] tensor
        """
        d_close = self.d_close
        d_far = self.d_far
        rfov = self.lens.rfov

        # In each iteration, sample one focus distance, [mm], range [foc_d_far, foc_d_close]
        # Example beta distribution: https://share.google/images/Mrb9c39PdddYx3UHj
        beta_sample = float(np.random.beta(1, 4))  # Biased towards 0
        foc_dist = self.foc_d_close + beta_sample * (self.foc_d_far - self.foc_d_close)
        self.lens.refocus(foc_dist)
        foc_dist = torch.full((num_points,), foc_dist)

        # Sample (fov), [radians], range[0, rfov]
        beta_values = np.random.beta(4, 1, num_points)  # Biased towards 1
        beta_values = torch.from_numpy(beta_values).float()
        fov = beta_values * rfov

        # Sample (depth), sample more points near the focus distance, [mm], range [d_far, d_close]
        # A smaller std_dev value samples points more tightly
        std_dev = -foc_dist / concentration_factor
        depth = foc_dist + torch.randn(num_points) * std_dev
        depth = torch.clamp(depth, d_far, d_close)

        # Create input tensor
        sample_input = torch.stack([fov, depth / 1000.0, foc_dist / 1000.0], dim=1)
        sample_input = sample_input.to(self.device)

        # Calculate PSF by ray tracing, shape of [B, 3, ks, ks]
        points_x = torch.zeros_like(depth)
        points_y = self.lens.foclen * torch.tan(fov) / self.lens.r_sensor
        points_z = depth
        points = torch.stack((points_x, points_y, points_z), dim=-1)
        sample_psf = self.lens.psf_rgb(
            points=points, ks=self.kernel_size, recenter=True
        )

        return sample_input, sample_psf

    def eval(self):
        """Set the network to evaluation mode."""
        self.psfnet.eval()

    def points2input(self, points):
        """Convert points to input tensor.

        Args:
            points (tensor): [N, 3] tensor, [-1, 1] * [-1, 1] * [depth_min, depth_max]

        Returns:
            input (tensor): [N, 3] tensor, (fov, depth, foc_dist).
                - fov from [0, rfov] on y-axis, [radians]
                - depth/1000.0 from [d_far, d_close], [mm]
                - foc_dist/1000.0 from [foc_d_far, foc_d_close], [mm]
        """
        sensor_h, sensor_w = self.lens.sensor_size
        foclen = self.lens.foclen

        points_x = points[:, 0] * sensor_w / 2
        points_y = points[:, 1] * sensor_h / 2
        points_r = torch.sqrt(points_x**2 + points_y**2)
        fov = torch.atan(points_r / foclen)
        depth = points[:, 2]
        foc_dist = torch.full_like(fov, self.foc_dist)
        network_inp = torch.stack((fov, depth / 1000.0, foc_dist / 1000.0), dim=-1)
        return network_inp

    # ==================================================
    # Network inference
    # ==================================================
    def refocus(self, foc_dist):
        """Refocus the lens to the given focus distance."""
        self.lens.refocus(foc_dist)
        self.foc_dist = foc_dist

    def psf_rgb(self, points, ks=64):
        """Calculate RGB PSF using the PSF network.

        Args:
            points (tensor): [N, 3] tensor, [-1, 1] * [-1, 1] * [depth_min, depth_max]
            foc_dist (float): focus distance

        Returns:
            psf (tensor): [N, 3, ks, ks] tensor
        """
        # Calculate network input
        network_inp = self.points2input(points)

        # Predict y-axis PSF from network
        psf = self.psfnet(network_inp)

        # Post-process PSF
        # The psfnet is trained with PSFs on the y-axis.
        # We need to rotate the PSF to the correct orientation based on the point's coordinates.
        # The counter-clockwise angle from the positive y-axis to the point (x, y) is atan2(x, y).
        rot_angle = torch.atan2(points[:, 0], points[:, 1])
        psf = rotate_psf(psf, rot_angle)

        # Crop PSF to the given kernel size
        if ks < self.kernel_size:
            psf = psf[
                :,
                :,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
            ]
        return psf

    def psf_map_rgb(self, grid=(11, 11), depth=DEPTH, ks=PSF_KS, **kwargs):
        """Compute monochrome PSF map.

        Args:
            grid (tuple, optional): Grid size. Defaults to (11, 11), meaning 11x11 grid.
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            ks (int, optional): Kernel size. Defaults to PSF_KS, meaning PSF_KS x PSF_KS kernel size.

        Returns:
            psf_map: Shape of [grid, grid, 3, ks, ks].
        """
        # PSF map grid
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3).to(self.device)

        # Compute PSF map
        psf = self.psf_rgb(points=points, ks=ks)  # [grid*grid, 3, ks, ks]
        psf_map = psf.reshape(grid[0], grid[1], 3, ks, ks)  # [grid, grid, 3, ks, ks]
        return psf_map

    # ==================================================
    # Image simulation
    # ==================================================
    @torch.no_grad()
    def render_rgbd(self, img, depth, foc_dist, ks=64, high_res=False):
        """Render image with aif image and depth map. Receive [N, C, H, W] image.

        Args:
            img (tensor): [1, C, H, W]
            depth (tensor): [1, H, W], depth map, unit in mm, range from [-20000, -200]
            foc_dist (tensor): [1], unit in mm, range from [-20000, -200]
            ks (int): kernel size
            high_res (bool): whether to use high resolution rendering

        Returns:
            render (tensor): [1, C, H, W]
        """
        B, C, H, W = img.shape
        assert B == 1, "Only support batch size 1"

        # Refocus the lens to the given focus distance
        self.refocus(foc_dist)

        # Estimate PSF for each pixel
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=self.device),
            torch.linspace(1, -1, H, device=self.device),
            indexing="xy",
        )
        x, y = x.unsqueeze(0).repeat(B, 1, 1), y.unsqueeze(0).repeat(B, 1, 1)
        depth = depth.squeeze(1) / 1000.0

        points = torch.stack((x, y, depth), -1).float()
        psf = self.psf_rgb(points=points, ks=ks)

        # Render image with per-pixel PSF convolution
        if high_res:
            render = conv_psf_pixel_high_res(img, psf)
        else:
            render = conv_psf_pixel(img, psf)

        return render

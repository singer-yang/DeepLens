# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""A surrogate network model to represent the PSF of a lens.

Technical Paper:
    Xinge Yang, Qiang Fu, Mohamed Elhoseiny, and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deeplens.lens import Lens
from deeplens.geolens import GeoLens
from deeplens.network.surrogate import MLP, PSFNet_MLPConv
from deeplens.network.surrogate.psfnet_mplconv2 import PSFNet_MLPConv2
from deeplens.network.surrogate.psfnet_mplconv3 import PSFNet_MLPConv3
from deeplens.optics.psf import conv_psf_pixel, conv_psf_pixel_high_res, rotate_psf
from deeplens.optics.basics import DEPTH


class PSFNetLens(Lens):
    def __init__(
        self,
        lens_path,
        in_chan=3,
        psf_chan=3,
        model_name="mlp_conv",
        kernel_size=64,
        sensor_res=(2000, 3000),
    ):
        super().__init__()

        # Load lens
        self.lens_path = lens_path
        self.lens = GeoLens(filename=lens_path, device=self.device)
        self.lens.set_sensor_res(sensor_res=sensor_res)
        self.hfov = self.lens.hfov

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

        print(
            f"Sensor pixel size is {self.lens.pixel_size * 1000} um, PSF kernel size is {self.kernel_size}."
        )

        # There is a minimum focal distance for each lens.
        # For example, the Canon EF 50mm lens can only focus to 0.5m and further.
        self.foc_d_min = -500
        self.foc_d_max = -20000
        self.foc_d_arr = (np.linspace(0, 1, 100)) ** 2 * (
            self.foc_d_max - self.foc_d_min
        ) + self.foc_d_min
        self.foc_d_arr = np.round(self.foc_d_arr, 0)

        # depth range
        self.d_min = -200
        self.d_max = -20000

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
        elif model_name == "mlpconv2":
            psfnet = PSFNet_MLPConv2(
                in_chan=in_chan, kernel_size=kernel_size, out_chan=psf_chan
            )
        elif model_name == "mlpconv3":
            psfnet = PSFNet_MLPConv3(
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
        psfnet_dict = torch.load(net_path, map_location="cpu", weights_only=False)
        assert psfnet_dict["pixel_size"] == self.pixel_size, (
            "Pixel size mismatch between network and lens"
        )
        assert psfnet_dict["lens_path"] == self.lens_path, (
            "Lens path mismatch between network and lens"
        )
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
        lr=5e-4,
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

        self.save_psfnet(f"{result_dir}/PSFNet_{self.model_name}.pth")

    def sample_training_data(self, num_points=512, concentration_factor=2.0):
        """Sample training data for PSF surrogate network.

        Args:
            num_points (int): number of training points
            concentration_factor (float): concentration factor for training data sampling

        Returns:
            sample_input (tensor): [B, 3] tensor, (fov, depth, foc_dist).
                - fov from [0, hfov] on 0y-axis, [radians]
                - depth from [d_min, d_max], [mm]
                - foc_dist from [foc_d_min, foc_d_max], [mm]
                - We use absolute fov and depth.

            sample_psf (tensor): [B, 3, ks, ks] tensor
        """
        d_min = self.d_min
        d_max = self.d_max
        eff_hfov = self.lens.calc_eff_hfov()

        # In each iteration, sample one focus distance, [mm], range [foc_d_max, foc_d_min]
        foc_dist = float(np.random.choice(self.foc_d_arr))
        self.lens.refocus(foc_dist)

        # Sample (fov), uniform distribution, [radians], range[0, hfov]
        fov = torch.rand(num_points) * eff_hfov

        # Sample (depth), sample more points near the focus distance, [mm], range [d_max, d_min]
        std_dev = (
            -foc_dist / concentration_factor
        )  # A smaller value concentrates points more tightly. Default is -foc_dist / 2.0
        depth = foc_dist + torch.randn(num_points) * std_dev
        depth = torch.clamp(depth, d_max, d_min)

        # Create input tensor
        sample_input = torch.stack(
            [fov, depth / 1000.0, torch.full((num_points,), foc_dist / 1000.0)], dim=1
        )
        sample_input = sample_input.to(self.device)

        # Calculate PSF by ray tracing, shape of [B, 3, ks, ks]
        points_x = torch.zeros_like(depth)
        points_y = self.lens.foclen * torch.tan(fov) / self.lens.r_sensor
        points_z = depth
        points = torch.stack((points_x, points_y, points_z), dim=-1)
        with torch.no_grad():
            sample_psf = self.lens.psf_rgb(
                points=points, ks=self.kernel_size, recenter=True
            )

        return sample_input, sample_psf

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
        sensor_h, sensor_w = self.lens.sensor_size
        points_x = points[:, 0] * sensor_w / 2
        points_y = points[:, 1] * sensor_h / 2
        foclen = self.lens.foclen
        points_fov = torch.atan(torch.sqrt(points_x**2 + points_y**2) / foclen)
        foc_dist = torch.full_like(points_fov, self.foc_dist)
        network_inp = torch.stack(
            (points_fov, points[:, 2] / 1000.0, foc_dist / 1000.0), dim=-1
        )

        # Predict 0y-axis PSF from network
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

    def psf_map_rgb(self, grid=(11, 11), depth=DEPTH, ks=51, **kwargs):
        """Compute monochrome PSF map.

        Args:
            grid (tuple, optional): Grid size. Defaults to (5, 5), meaning 5x5 grid.
            wvln (float, optional): Wavelength. Defaults to 0.589.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            ks (int, optional): Kernel size. Defaults to 51, meaning 51x51 kernel size.

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

    @torch.no_grad()
    def render_rgbd(self, img, depth, foc_dist, high_res=False):
        """Render image with aif image and depth map. Receive [N, C, H, W] image.

        Args:
            img (tensor): [1, C, H, W]
            depth (tensor): [1, H, W], depth map, unit in mm, range from [-20000, -200]
            foc_dist (tensor): [1], unit in mm, range from [-20000, -200]
            high_res (bool): whether to use high resolution rendering

        Returns:
            render (tensor): [1, C, H, W]
        """
        B, C, H, W = img.shape
        assert B == 1, "Only support batch size 1"

        # Estimate PSF for each pixel
        z = self.depth2z(depth).squeeze(1)
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, W), torch.linspace(1, -1, H), indexing="xy"
        )
        x, y = x.unsqueeze(0).repeat(B, 1, 1), y.unsqueeze(0).repeat(B, 1, 1)
        x, y = x.to(img.device), y.to(img.device)

        # =====>
        # foc_dist = foc_dist.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
        # foc_z = self.depth2z(foc_dist)
        # o = torch.stack((x, y, z, foc_z), -1).float()
        # psf = self.pred_psf(o)
        # =====>
        o = torch.stack((x, y, z), -1).float()
        self.refocus(
            foc_dist.item() if isinstance(foc_dist, torch.Tensor) else foc_dist
        )
        psf = self.psf_rgb(points=o)
        # =====>

        # Render image with per-pixel PSF convolution
        if high_res:
            render = conv_psf_pixel_high_res(img, psf)
        else:
            render = conv_psf_pixel(img, psf)

        return render


# ==================================================================
# Thin lens model (baseline)
# ==================================================================
# class ThinLens(DeepObj):
#     def __init__(
#         self, foc_len, fnum, kernel_size, sensor_size, sensor_res, device="cpu"
#     ):
#         super(ThinLens, self).__init__()

#         self.d_max = -20000
#         self.d_min = -200
#         self.kernel_size = kernel_size
#         self.foc_len = foc_len
#         self.fnum = fnum
#         self.sensor_size = sensor_size
#         self.sensor_res = sensor_res
#         self.ps = self.sensor_size[0] / self.sensor_res[0]

#     def coc(self, depth, foc_dist):
#         if (depth < 0).any():
#             depth = -depth
#             foc_dist = -foc_dist

#         depth = torch.clamp(depth, self.d_min, self.d_max)
#         coc = (
#             self.foc_len
#             / self.fnum
#             * torch.abs(depth - foc_dist)
#             / depth
#             * self.foc_len
#             / (foc_dist - self.foc_len)
#         )
#         # clamp_min is only a random constant avoiding getting coc_pixel = 0
#         clamp_min = 2 if self.kernel_size % 2 == 0 else 0.2
#         coc_pixel = torch.clamp(coc / self.ps, min=clamp_min)
#         return coc_pixel

#     def render(self, img, depth, foc_dist, high_res=False):
#         """Render image with aif image and Gaussian PSFs.

#         Args:
#             img: [N, C, H, W]
#             depth: [N, 1, H, W]
#             foc_dist: [N]

#         Raises:
#             Exception: _description_

#         Returns:
#             _type_: _description_
#         """
#         ks = self.kernel_size
#         device = img.device

#         if len(img.shape) == 3:
#             raise Exception("Untested.")

#         elif len(img.shape) == 4:
#             N, C, H, W = img.shape

#             # [N] to [N, 1, H, W]
#             foc_dist = (
#                 foc_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
#             )

#             psf = torch.zeros((N, H, W, ks, ks), device=device)
#             x, y = torch.meshgrid(
#                 torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
#                 torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks),
#                 indexing="xy",
#             )
#             x, y = x.to(device), y.to(device)

#             coc_pixel = self.coc(depth, foc_dist)
#             # Shape expands to [N, H, W, ks, ks]
#             coc_pixel = (
#                 coc_pixel.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, ks, ks)
#             )
#             coc_pixel_radius = coc_pixel / 2
#             psf = torch.exp(-(x**2 + y**2) / 2 / coc_pixel_radius**2) / (
#                 2 * np.pi * coc_pixel_radius**2
#             )
#             psf_mask = x**2 + y**2 < coc_pixel_radius**2
#             psf = psf * psf_mask
#             psf = psf / psf.sum((-1, -2)).unsqueeze(-1).unsqueeze(-1)

#             render = conv_psf_pixel(img, psf)
#             return render

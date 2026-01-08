# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Paraxial diffractive lens model. Each optical element (lens, DOE, metasurface, etc.) in the paraxial diffractive model is modeled as a phase function. This simplified optical model is easy to use (but typically not accurate enough) for many real-world applications.

Reference papers:
    [1] Vincent Sitzmann*, Steven Diamond*, Yifan Peng*, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, Gordon Wetzstein, "End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging," Siggraph 2018.
    [2] Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide. "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range Imaging," CVPR 2020.
"""

import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from deeplens.basics import DEFAULT_WAVE, DEPTH, PSF_KS
from deeplens.lens import Lens
from deeplens.optics.diffractive_surface import (
    Binary2,
    Fresnel,
    Pixel2D,
    ThinLens,
    Zernike,
)
from deeplens.optics.psf import conv_psf
from deeplens.optics.utils import diff_float
from deeplens.optics.wave import ComplexWave


class DiffractiveLens(Lens):
    def __init__(
        self,
        filename=None,
        device=None,
    ):
        """Initialize a diffractive lens.

        Args:
            filename (str, optional): Path to the lens configuration JSON file. If provided, loads the lens configuration from file. Defaults to None.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__(device=device)

        # Load lens file
        if filename is not None:
            self.read_lens_json(filename)
        else:
            self.surfaces = []
            # Set default sensor size and resolution if no file provided
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)

        self.double()

    @classmethod
    def load_example1(cls):
        """Create an example diffractive lens with a single Fresnel DOE.

        Returns:
            DiffractiveLens: A configured diffractive lens with a Fresnel surface
                at f=50mm, 4mm size, and 4000 resolution.
        """
        self = cls(sensor_size=(4.0, 4.0), sensor_res=(2000, 2000))

        # Diffractive Fresnel DOE
        self.surfaces = [Fresnel(f0=50, d=0, size=4, res=4000)]

        # Sensor
        self.d_sensor = torch.tensor(50)

        self.to(self.device)
        return self

    @classmethod
    def load_example2(cls):
        """Create an example diffractive lens with a thin lens and binary DOE combination.

        Returns:
            DiffractiveLens: A configured diffractive lens with a ThinLens (f=50mm)
                and a Binary2 DOE, both at 4mm size and 4000 resolution.
        """
        self = cls(sensor_size=(8.0, 8.0), sensor_res=(2000, 2000))

        # Diffractive Fresnel DOE
        self.surfaces = [
            ThinLens(f0=50, d=0, size=4, res=4000),
            Binary2(d=0, size=4, res=4000),
        ]

        # Sensor
        self.d_sensor = torch.tensor(50)
        self.sensor_size = (8.0, 8.0)
        self.sensor_res = (2000, 2000)

        self.to(self.device)
        return self

    def read_lens_json(self, filename):
        """Load the lens configuration from a JSON file.

        Reads lens parameters including sensor configuration and diffractive surfaces
        from the specified JSON file. If sensor_size or sensor_res are not provided,
        defaults of 8mm x 8mm and 2000x2000 pixels will be used.

        Args:
            filename (str): Path to the JSON configuration file.
        """
        assert filename.endswith(".json"), "File must be a .json file."

        with open(filename, "r") as f:
            # Lens general info
            data = json.load(f)
            self.d_sensor = torch.tensor(data["d_sensor"])
            self.lens_info = data.get("info", "None")

            # Read sensor_size with default
            if "sensor_size" in data:
                self.sensor_size = tuple(data["sensor_size"])
            else:
                self.sensor_size = (8.0, 8.0)
                print(
                    f"Sensor_size not found in lens file. Using default: {self.sensor_size} mm. "
                    "Consider specifying sensor_size in the lens file or using set_sensor()."
                )

            # Read sensor_res with default
            if "sensor_res" in data:
                self.sensor_res = tuple(data["sensor_res"])
            else:
                self.sensor_res = (2000, 2000)
                print(
                    f"Sensor_res not found in lens file. Using default: {self.sensor_res} pixels. "
                    "Consider specifying sensor_res in the lens file or using set_sensor()."
                )

            # Load diffractive surfaces/elements
            d = 0.0
            self.surfaces = []
            for surf_dict in data["surfaces"]:
                surf_dict["d"] = d

                if surf_dict["type"].lower() == "binary2":
                    s = Binary2.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "fresnel":
                    s = Fresnel.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "pixel2d":
                    s = Pixel2D.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "thinlens":
                    s = ThinLens.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "zernike":
                    s = Zernike.init_from_dict(surf_dict)
                else:
                    raise ValueError(
                        f"Diffractive surface type {surf_dict['type']} not implemented."
                    )

                self.surfaces.append(s)
                d_next = surf_dict["d_next"]
                d += d_next

    def write_lens_json(self, filename):
        """Write the lens configuration to a JSON file.

        Saves all lens parameters including sensor configuration and
        diffractive surface data to the specified file.

        Args:
            filename (str): Output path for the JSON file.
        """
        assert filename.endswith(".json"), "File must be a .json file."

        # Save lens to a file
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        data["d_sensor"] = round(self.d_sensor.item(), 3)
        data["l_sensor"] = round(self.l_sensor, 3)
        data["sensor_res"] = self.sensor_res

        # Save diffractive surfaces
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i + 1}

            if isinstance(s, Pixel2D):
                surf_data = s.surf_dict(filename.replace(".json", "_pixel2d.pth"))
            else:
                surf_data = s.surf_dict()

            surf_dict.update(surf_data)

            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = (
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item()
                )

            data["surfaces"].append(surf_dict)

        # Save data to a file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    # =============================================
    # Utils
    # =============================================
    def __call__(self, wave):
        """Propagate a wave through the lens system."""
        return self.forward(wave)

    def forward(self, wave):
        """Propagate a wave through the diffractive lens system to the sensor.

        Sequentially applies phase modulation from each diffractive surface, then propagates
        the wave to the sensor plane using wave optics.

        Args:
            wave (ComplexWave): Input wave field entering the lens system.

        Returns:
            ComplexWave: Output wave field at the sensor plane.
        """
        # Propagate to DOE
        for surf in self.surfaces:
            wave = surf(wave)

        # Propagate to sensor
        wave = wave.prop_to(self.d_sensor.item())

        return wave

    # =============================================
    # Image simulation
    # =============================================
    def render_mono(self, img, wvln=DEFAULT_WAVE, ks=PSF_KS):
        """Simulate monochromatic lens blur by convolving an image with the point spread function.

        Args:
            img (torch.Tensor): Input image. Shape: (B, 1, H, W)
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.
            ks (int, optional): PSF kernel size. Defaults to PSF_KS.

        Returns:
            torch.Tensor: Rendered image after applying lens blur with shape (B, 1, H, W).
        """
        psf = self.psf_infinite(wvln=wvln, ks=ks).unsqueeze(0)  # (1, ks, ks)
        img_render = conv_psf(img, psf)
        return img_render

    def psf(self, depth=float("inf"), wvln=DEFAULT_WAVE, ks=PSF_KS, upsample_factor=1):
        """Calculate monochromatic point PSF by wave propagation approach.

        Args:
            depth (float, optional): Depth of the point source. Defaults to float('inf').
            wvln (float, optional): Wavelength in micrometers. Defaults to DEFAULT_WAVE.
            ks (int, optional): PSF kernel size. Defaults to PSF_KS.
            upsample_factor (int, optional): Upsampling factor to meet Nyquist sampling constraint. Defaults to 1.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]

        Note:
            [1] Usually we only consider the on-axis PSF because paraxial approximation is implicitly applied for wave optical model. For the shifted phase issue, refer to "Modeling off-axis diffraction with the least-sampling angular spectrum method".
        """
        # Sample input wave field (We have to sample high resolution to meet Nyquist sampling constraint)
        field_res = [
            self.surfaces[0].res[0] * upsample_factor,
            self.surfaces[0].res[1] * upsample_factor,
        ]
        field_size = [
            self.surfaces[0].res[0] * self.surfaces[0].ps,
            self.surfaces[0].res[1] * self.surfaces[0].ps,
        ]
        if depth == float("inf"):
            inp_wave = ComplexWave.plane_wave(
                phy_size=field_size,
                res=field_res,
                wvln=wvln,
                z=0.0,
            ).to(self.device)
        else:
            inp_wave = ComplexWave.point_wave(
                point=[0.0, 0.0, depth],
                phy_size=field_size,
                res=field_res,
                wvln=wvln,
                z=0.0,
            ).to(self.device)

        # Calculate intensity on the sensor. Shape [H_sensor, W_sensor]
        output_wave = self.forward(inp_wave)
        intensity = output_wave.u.abs() ** 2

        # Interpolate wave to have the same pixel size as the sensor
        factor = output_wave.ps / self.pixel_size
        intensity = F.interpolate(
            intensity,
            scale_factor=(factor, factor),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]

        # Crop or pad wave to the sensor resolution
        intensity_h, intensity_w = intensity.shape[-2:]
        sensor_h, sensor_w = self.sensor_res
        if sensor_h < intensity_h or sensor_w < intensity_w:
            # crop
            start_h = (intensity_h - sensor_h) // 2
            start_w = (intensity_w - sensor_w) // 2
            intensity = intensity[
                start_h : start_h + sensor_h, start_w : start_w + sensor_w
            ]
        elif sensor_h > intensity_h or sensor_w > intensity_w:
            # pad
            pad_top = (sensor_h - intensity_h) // 2
            pad_bottom = sensor_h - intensity_h - pad_top
            pad_left = (sensor_w - intensity_w) // 2
            pad_right = sensor_w - intensity_w - pad_left
            intensity = F.pad(
                intensity,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

        # Crop the valid patch from the full-resolution intensity map as the PSF
        coord_c_i = int(self.sensor_res[1] / 2)
        coord_c_j = int(self.sensor_res[0] / 2)
        intensity = F.pad(
            intensity,
            [ks // 2, ks // 2, ks // 2, ks // 2],
            mode="constant",
            value=0,
        )
        psf = intensity[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]

        # Normalize PSF
        psf /= psf.sum()
        psf = torch.flip(psf, [0, 1])

        return diff_float(psf)

    # =============================================
    # Visualization
    # =============================================
    def draw_layout(self, save_name="./doelens.png"):
        """Draw the lens layout diagram.

        Visualizes the DOE and sensor positions in a 2D layout.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to './doelens.png'.
        """
        fig, ax = plt.subplots()

        # Draw DOE
        d = self.doe.d.item()
        doe_l = self.doe.l
        ax.plot(
            [d, d], [-doe_l / 2, doe_l / 2], "orange", linestyle="--", dashes=[1, 1]
        )

        # Draw sensor
        d = self.sensor.d.item()
        sensor_l = self.sensor.l
        width = 0.2  # Width of the rectangle
        rect = plt.Rectangle(
            (d - width / 2, -sensor_l / 2),
            width,
            sensor_l,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_psf(
        self,
        depth=DEPTH,
        ks=PSF_KS,
        save_name="./psf_doelens.png",
        log_scale=True,
        eps=1e-4,
    ):
        """Draw on-axis RGB PSF.

        Computes and saves a visualization of the RGB PSF for a given depth.

        Args:
            depth (float, optional): Depth of the point source. Defaults to DEPTH.
            ks (int, optional): Size of the PSF kernel in pixels. Defaults to PSF_KS.
            save_name (str, optional): Path to save the PSF image. Defaults to './psf_doelens.png'.
            log_scale (bool, optional): If True, display PSF in log scale. Defaults to True.
            eps (float, optional): Small value for log scale to avoid log(0). Defaults to 1e-4.
        """
        psf_rgb = self.psf_rgb(point=[0, 0, depth], ks=ks)

        if log_scale:
            psf_rgb = torch.log10(psf_rgb + eps)
            psf_rgb = (psf_rgb - psf_rgb.min()) / (psf_rgb.max() - psf_rgb.min())
            save_name = save_name.replace(".png", "_log.png")

        save_image(psf_rgb.unsqueeze(0), save_name, normalize=True)

    # =============================================
    # Optimization
    # =============================================
    def get_optimizer(self, lr):
        """Get optimizer for the lens parameters.

        Args:
            lr (float): Learning rate.

        Returns:
            Optimizer: Optimizer object for lens parameters.
        """
        return self.doe.get_optimizer(lr=lr)

# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Paraxial diffractive lens model consisting of a diffractive optical element (DOE) and a sensor.

Reference papers:
    [1] Vincent Sitzmann*, Steven Diamond*, Yifan Peng*, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, Gordon Wetzstein, "End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging," Siggraph 2018.
    [2] Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide. "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range Imaging," CVPR 2020.
"""

import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from deeplens.lens import Lens
from deeplens.optics.basics import DEPTH, DEFAULT_WAVE
from deeplens.optics.diffractive_surface import Binary2, Fresnel, Pixel2D, ThinLens, Zernike
from deeplens.optics.waveoptics_utils import point_source_field, plane_wave_field
from deeplens.optics.render_psf import render_psf
from deeplens.optics.utils import diff_float

class DiffractiveLens(Lens):
    def __init__(
        self,
        filename=None,
        sensor_res=(2000, 2000),
        sensor_size=(8.0, 8.0),
        device=None,
    ):
        """Initialize a lens consisting of a diffractive optical element (DOE).

        Args:
            filename (str): Path to the lens file.
            sensor_res (tuple, optional): Sensor resolution (W, H). Defaults to (2000, 2000).
            sensor_size (tuple, optional): Sensor size (W, H). Defaults to (8.0, 8.0).
            device (str, optional): Device to run the lens. Defaults to "cpu".
        """
        super().__init__(device=device)

        # Lens sensor size and resolution
        self.sensor_res = sensor_res
        self.sensor_size = sensor_size

        # Load lens file
        if filename is not None:
            self.read_lens_json(filename)
        else:
            self.surfaces = []

        self.double()

    @classmethod
    def load_example1(cls):
        self = cls(sensor_size=(4.0, 4.0), sensor_res=(2000, 2000))

        # Diffractive Fresnel DOE
        self.surfaces = [Fresnel(f0=50, d=0, size=4, res=4000)]

        # Sensor
        self.d_sensor = torch.tensor(50)

        self.to(self.device)
        return self

    @classmethod
    def load_example2(cls):
        """Initialize a lens from a dict."""
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
        """Load lens from a .json file."""
        assert filename.endswith(".json"), "File must be a .json file."

        with open(filename, "r") as f:
            # Lens general info
            data = json.load(f)
            self.d_sensor = torch.tensor(data["d_sensor"])
            self.sensor_size = data["sensor_size"]
            self.sensor_res = data["sensor_res"]
            self.lens_info = data["info"]

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
        """Write the lens into a file."""
        assert filename.endswith(".json"), "File must be a .json file."

        # Save lens to a file
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        data["d_sensor"] = round(self.d_sensor.item(), 2)
        data["l_sensor"] = round(self.l_sensor, 2)
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
        return self.forward(wave)

    def forward(self, wave):
        """Propagate a wave through the optical element.

        Args:
            wave (Wave): Input wave field.

        Returns:
            wave (Wave): Output wave field at sensor plane.
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
    def render_mono(self, img, wvln=DEFAULT_WAVE, ks=101):
        """Apply PSF to simulate lens blur for single spectral channel image.

        Args:
            img (torch.Tensor): Input image. Shape: (B, 1, H, W)
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.
            ks (int, optional): PSF kernel size. Defaults to 101.

        Returns:
            img_render (torch.Tensor): Rendered image. Shape: (B, C, H, W)
        """
        psf = self.psf_infinite(wvln=wvln, ks=ks).unsqueeze(0)  # (1, ks, ks)
        img_render = render_psf(img, psf)
        return img_render

    def psf(self, depth=float("inf"), wvln=0.589, ks=101, upsample_factor=1):
        """Calculate monochromatic point PSF by wave propagation approach.

        Args:
            depth (float, optional): Depth of the point source. Defaults to float('inf').
            wvln (float, optional): wvln. Defaults to 0.589 [um].
            ks (int, optional): PSF kernel size. Defaults to 101.
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
            inp_wave = plane_wave_field(
                phy_size=field_size,
                res=field_res,
                wvln=wvln,
                z=0.0,
            ).to(self.device)
        else:
            inp_wave = point_source_field(
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
            intensity = intensity[start_h : start_h + sensor_h, start_w : start_w + sensor_w]
        elif sensor_h > intensity_h or sensor_w > intensity_w:
            # pad
            pad_top = (sensor_h - intensity_h) // 2
            pad_bottom = sensor_h - intensity_h - pad_top
            pad_left = (sensor_w - intensity_w) // 2
            pad_right = sensor_w - intensity_w - pad_left
            intensity = F.pad(intensity, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

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
        """Draw lens setup."""
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
        ks=101,
        save_name="./psf_doelens.png",
        log_scale=True,
        eps=1e-4,
    ):
        """Draw on-axis RGB PSF.

        Args:
            depth (float): Depth of the point source
            ks (int): Size of the PSF kernel
            save_name (str): Path to save the PSF image
            log_scale (bool): If True, display PSF in log scale
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
        return self.doe.get_optimizer(lr=lr)

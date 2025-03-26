"""Paraxial diffractive lens model consisting of a diffractive optical element (DOE) and a sensor.


Reference papers:
    [1] Vincent Sitzmann*, Steven Diamond*, Yifan Peng*, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, Gordon Wetzstein, "End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging," Siggraph 2018.
    [2] Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide. "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range Imaging," CVPR 2020.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .lens import Lens
from .optics.basics import DEPTH, EPSILON, DEFAULT_WAVE
from .optics.diffractive_surface import Binary2, Fresnel, Pixel2D, ThinLens, Zernike
from .optics.materials import Material
from .optics.waveoptics_utils import point_source_field, plane_wave_field


class DiffractiveLens(Lens):
    def __init__(self, filename=None, sensor_res=(2000, 2000), device=None):
        """Initialize a lens consisting of a diffractive optical element (DOE).

        Args:
            filename (str): Path to the lens file.
            sensor_res (tuple, optional): Sensor resolution (W, H). Defaults to (2000, 2000).
            device (str, optional): Device to run the lens. Defaults to "cpu".
        """
        super().__init__(lens_path=filename, sensor_res=sensor_res, device=device)

        torch.set_default_dtype(torch.float64)
        self.double()

    @classmethod
    def load_example1(cls):
        self = cls(sensor_res=(2000, 2000))

        # Diffractive Fresnel DOE
        self.surfaces = [Fresnel(f0=50, d=0, size=4, res=4000)]

        # Sensor
        self.d_sensor = torch.tensor(50)
        self.l_sensor = 4

        self.to(self.device)
        return self

    @classmethod
    def load_example2(cls):
        """Initialize a lens from a dict."""
        self = cls(sensor_res=(2000, 2000))

        # Diffractive Fresnel DOE
        self.surfaces = [
            ThinLens(f0=50, d=0, size=4, res=4000),
            Binary2(d=0, size=4, res=4000),
        ]

        # Sensor
        self.d_sensor = torch.tensor(50)
        self.l_sensor = 4

        self.to(self.device)
        return self

    def read_lens_json(self, filename):
        """Load lens from a .json sfile."""
        assert filename.endswith(".json"), "File must be a .json file."

        with open(filename, "r") as f:
            # Lens general info
            data = json.load(f)
            self.d_sensor = data["d_sensor"]
            self.l_sensor = data["l_sensor"]
            self.sensor_res = data["sensor_res"]
            self.lens_info = data["info"]

            # Load diffractive surfaces/elements
            d = 0.0
            self.surfaces = []
            for surf_dict in data["surfaces"]:
                surf_dict["d"] = d

                if surf_dict["type"] == "Binary2":
                    s = Binary2.init_from_dict(surf_dict)
                elif surf_dict["type"] == "Fresnel":
                    s = Fresnel.init_from_dict(surf_dict)
                elif surf_dict["type"] == "Pixel2D":
                    s = Pixel2D.init_from_dict(surf_dict)
                elif surf_dict["type"] == "ThinLens":
                    s = ThinLens.init_from_dict(surf_dict)
                elif surf_dict["type"] == "Zernike":
                    s = Zernike.init_from_dict(surf_dict)
                else:
                    raise ValueError(
                        f"Diffractive surface type {surf_dict['type']} not implemented."
                    )

                self.surfaces.append(s)
                d_next = surf_dict["d_next"]
                d += d_next

    def forward(self, wave):
        """Propagate a wave through the optical element.

        Args:
            wave (Wave): Input wavefront.

        Returns:
            wave (Wave): Output wavefront.
        """
        # Propagate to DOE
        for surf in self.surfaces:
            wave = surf(wave)

        # Propagate to sensor
        wave = wave.prop_to(self.d_sensor.item())  # TODO: check if we need .item()

        return wave

    def __call__(self, wave):
        return self.forward(wave)

    # =============================================
    # PSF-related functions
    # =============================================
    def psf(self, point=[0, 0, -10000.0], wvln=0.589, ks=101):
        """Calculate monochromatic point PSF by wave propagation approach.

            For the shifted phase issue, refer to "Modeling off-axis diffraction with the least-sampling angular spectrum method".

        Args:
            point (list, optional): Normalized point source position [-1, 1] x [-1, 1] x [0, Inf]. Defaults to [0, 0, -10000.0].
            wvln (float, optional): wvln. Defaults to 0.589 [um].
            ks (int, optional): PSF kernel size. Defaults to 101.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]
        """
        # Get input wave field
        x, y, z = point
        sensor_l = self.l_sensor
        field_res = self.surfaces[0].res
        scale = -z / self.d_sensor.item()
        x_obj, y_obj = x * scale * sensor_l / 2, y * scale * sensor_l / 2

        # We have to sample high resolution to meet Nyquist sampling constraint.
        inp_wave = point_source_field(
            point=[x_obj, y_obj, z],
            phy_size=[sensor_l, sensor_l],
            res=field_res,
            wvln=wvln,
            fieldz=self.surfaces[0].d.item(),
            device=self.device,
        )

        # Calculate intensity on the sensor. Shape [H_sensor, W_sensor]
        output_wave = self.forward(inp_wave)
        intensity_full_res = output_wave.u.abs() ** 2
        intensity_full_res = F.interpolate(
            intensity_full_res,
            size=self.sensor_res,
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]

        # Crop the valid patch of the full-resolution intensity as the PSF
        coord_c_i = int((1 + y) * self.sensor_res[1] / 2)
        coord_c_j = int((1 - x) * self.sensor_res[0] / 2)
        intensity_full_res = F.pad(
            intensity_full_res,
            [ks // 2, ks // 2, ks // 2, ks // 2],
            mode="constant",
            value=0,
        )
        psf = intensity_full_res[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]

        # Normalize PSF
        psf /= psf.sum()
        psf = torch.flip(psf, [0, 1])

        return psf

    def psf_infinite(self, wvln=DEFAULT_WAVE, ks=101):
        """Calculate monochromatic PSF of infinite point source (plane wave).

        Args:
            wvln (float, optional): wvln. Defaults to 0.589 [um].
            ks (int, optional): PSF kernel size. Defaults to 101.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]
        """
        # Get input wave field
        sensor_l = self.l_sensor
        field_res = self.surfaces[0].res
        inp_wave = plane_wave_field(
            phy_size=[sensor_l, sensor_l], res=field_res, wvln=wvln, z=0.0, device=self.device
        )

        # Calculate intensity on the sensor. Shape [H_sensor, W_sensor]
        output_wave = self.forward(inp_wave)
        intensity_full_res = output_wave.u.abs() ** 2
        intensity_full_res = F.interpolate(
            intensity_full_res,
            size=self.sensor_res,
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]

        # Crop the valid patch of the full-resolution intensity as the PSF
        coord_c_i = self.sensor_res[1] // 2
        coord_c_j = self.sensor_res[0] // 2
        intensity_full_res = F.pad(
            intensity_full_res,
            [ks // 2, ks // 2, ks // 2, ks // 2],
            mode="constant",
            value=0,
        )
        psf = intensity_full_res[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]

        # Normalize PSF
        psf /= psf.sum()
        psf = torch.flip(psf, [0, 1])

        return psf

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

    # =============================================
    # IO
    # =============================================
    def write_file(self, filename):
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

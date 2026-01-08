# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Ray-wave model for hybrid refractive-diffractive lens. A hybrid lens consists of a GeoLens and a DOE in the back. A differentiable ray-wave model is used for optical simulation: first calculating the complex wavefield at the DOE plane by coherent ray tracing, then propagating the wavefield to the sensor plane by angular spectrum method. This hybrid lens model can simulate: (1) GeoLens aberration, and (2) DOE phase modulation.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu, Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import json

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deeplens.basics import (
    DEFAULT_WAVE,
    PSF_KS,
    SPP_COHERENT,
    WAVE_RGB,
)
from deeplens.geolens import GeoLens
from deeplens.lens import Lens
from deeplens.optics.diffractive_surface import (
    Binary2,
    Fresnel,
    Grating,
    Pixel2D,
    Zernike,
)
from deeplens.optics.geometric_surface import Plane
from deeplens.optics.monte_carlo import forward_integral
from deeplens.optics.phase_surface import Phase
from deeplens.optics.utils import diff_float
from deeplens.optics.wave import AngularSpectrumMethod


class HybridLens(Lens):
    """A hybrid refractive-diffractive lens with a Geolens and a DOE at last.

    This differentiable hybrid lens model can similate:
        1. Aberration of the refractive lens
        2. DOE phase modulation
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float64,
    ):
        """Initialize a hybrid refractive-diffractive lens.

        Args:
            filename (str, optional): Path to the lens configuration JSON file. Defaults to None.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float64.
        """
        super().__init__(device=device, dtype=dtype)

        # Load lens file
        if filename is not None:
            self.read_lens_json(filename)
        else:
            self.geolens = None
            self.doe = None
            # Set default sensor size and resolution if no file provided
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            print(
                f"No lens file provided. Using default sensor_size: {self.sensor_size} mm, "
                f"sensor_res: {self.sensor_res} pixels. Use set_sensor() to change."
            )

        self.double()

    def read_lens_json(self, filename):
        """Read the lens configuration from a JSON file.

        Loads a GeoLens and associated DOE from the specified file.
        Supported DOE types: binary2, pixel2d, fresnel, zernike, grating.

        Args:
            filename (str): Path to the JSON configuration file.
        """
        # Load geolens
        geolens = GeoLens(filename=filename, device=self.device)

        # Load DOE (diffractive surface)
        with open(filename, "r") as f:
            data = json.load(f)

            doe_dict = data["DOE"]
            doe_param_model = doe_dict["type"].lower()
            if doe_param_model == "binary2":
                doe = Binary2.init_from_dict(doe_dict)
            elif doe_param_model == "pixel2d":
                doe = Pixel2D.init_from_dict(doe_dict)
            elif doe_param_model == "fresnel":
                doe = Fresnel.init_from_dict(doe_dict)
            elif doe_param_model == "zernike":
                doe = Zernike.init_from_dict(doe_dict)
            elif doe_param_model == "grating":
                doe = Grating.init_from_dict(doe_dict)
            else:
                raise ValueError(f"Unsupported DOE parameter model: {doe_param_model}")
            self.doe = doe

        # Add a Plane/Phase surface to GeoLens (DOE placeholder)
        r_doe = float(np.sqrt(doe.w**2 + doe.h**2) / 2)
        geolens.surfaces.append(Plane(d=doe.d.item(), r=r_doe, mat2="air"))
        # r_doe = float(np.sqrt(doe.w**2 + doe.h**2) / 2)
        # geolens.surfaces.append(Phase(r=r_doe, d=doe.d))
        self.geolens = geolens
        self.foclen = geolens.foclen

        # Update hybrid lens sensor resolution and pixel size
        self.set_sensor(sensor_size=geolens.sensor_size, sensor_res=geolens.sensor_res)
        self.to(self.device)

    def write_lens_json(self, lens_path):
        """Write the lens configuration to a JSON file.

        Saves the GeoLens and DOE configurations to a JSON file.

        Args:
            lens_path (str): Path for the output JSON file.
        """
        geolens = self.geolens
        data = {}
        data["info"] = geolens.lens_info if hasattr(geolens, "lens_info") else "None"
        data["foclen"] = round(geolens.foclen, 4)
        data["fnum"] = round(geolens.fnum, 4)
        data["r_sensor"] = round(geolens.r_sensor, 4)
        data["d_sensor"] = round(geolens.d_sensor.item(), 4)
        data["sensor_size"] = [round(i, 4) for i in geolens.sensor_size]
        data["sensor_res"] = geolens.sensor_res

        # Geolens
        data["surfaces"] = []
        for i, s in enumerate(geolens.surfaces[:-1]):
            surf_dict = s.surf_dict()

            # To exclude the last surface (DOE)
            if i < len(geolens.surfaces) - 2:
                surf_dict["d_next"] = round(
                    geolens.surfaces[i + 1].d.item() - geolens.surfaces[i].d.item(), 3
                )
            else:
                surf_dict["d_next"] = round(
                    geolens.d_sensor.item() - geolens.surfaces[i].d.item(), 3
                )

            data["surfaces"].append(surf_dict)

        # DOE
        data["DOE"] = self.doe.surf_dict()

        with open(lens_path, "w") as f:
            json.dump(data, f, indent=4)

    # =====================================================================
    # Utils
    # =====================================================================
    def analysis(self, save_name="./test.png"):
        """Perform lens analysis and save visualizations.

        Draws the lens layout and DOE phase map.

        Args:
            save_name (str, optional): Base path for saving analysis images. Defaults to './test.png'.
        """
        self.draw_layout(save_name=save_name)
        self.doe.draw_phase_map(save_name=f"{save_name}_doe.png")

    def double(self):
        """Convert lens to double precision (float64) for accurate phase calculations."""
        self.geolens.astype(torch.float64)
        self.doe.astype(torch.float64)

    def refocus(self, foc_dist):
        """Refocus the HybridLens to a given depth.

        Adjusts the GeoLens focus distance. The DOE is not moved because it is
        installed with the geolens as described in the Siggraph Asia 2024 paper.

        Args:
            foc_dist (float): Target focus distance.
        """
        self.geolens.refocus(foc_dist)

    def calc_scale(self, depth):
        """Calculate the scale factor for object-to-image mapping.

        Args:
            depth (float): Object depth distance.

        Returns:
            float: Scale factor (object height / image height).
        """
        return self.geolens.calc_scale(depth)

    # =====================================================================
    # PSF-related functions
    # =====================================================================
    def doe_field(self, point, wvln=DEFAULT_WAVE, spp=SPP_COHERENT):
        """Compute the complex wave field at DOE plane using coherent ray tracing. This function re-implements geolens.pupil_field() by changing the computation position from pupil plane to the last surface (DOE). The wavefront stores information of all diffraction orders.

        Args:
            point (torch.Tensor): Tensor of shape (3,) representing the point source position. Defaults to torch.tensor([0.0, 0.0, -10000.0]).
            wvln (float): Wavelength. Defaults to DEFAULT_WAVE.
            spp (int): Samples per pixel. Must be >= 1,000,000 for accurate simulation. Defaults to SPP_COHERENT.

        Returns:
            wavefront: Tensor of shape [H, W] representing the complex wavefront.
            psf_center: List containing the PSF center coordinates [x, y].
        """
        assert spp >= 1_000_000, (
            "Coherent ray tracing spp is too small, "
            "which may lead to inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase tracing."
        )

        geolens, doe = self.geolens, self.doe

        if point.dim() == 1:
            point = point.unsqueeze(0)
        point = point.to(self.device)

        # Calculate ray origin in the object space
        scale = geolens.calc_scale(point[:, 2].item())
        point_obj = point.clone()
        point_obj[:, 0] = point[:, 0] * scale * geolens.sensor_size[1] / 2
        point_obj[:, 1] = point[:, 1] * scale * geolens.sensor_size[0] / 2

        # Determine ray center via chief ray
        pointc_chief_ray = geolens.psf_center(point_obj, method="chief_ray")[
            0
        ]  # shape [2]

        # Ray tracing to the DOE plane
        ray = geolens.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)
        ray.coherent = True
        ray, _ = geolens.trace(ray)
        ray = ray.prop_to(doe.d)

        # Calculate full-resolution complex field for exit-pupil diffraction
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=doe.ps,
            ks=doe.res[0],
            pointc=torch.zeros_like(point[:, :2]),
        ).squeeze(0)  # shape [H, W]

        # Compute PSF center based on chief ray
        psf_center = [
            pointc_chief_ray[0] / geolens.sensor_size[0] * 2,
            pointc_chief_ray[1] / geolens.sensor_size[1] * 2,
        ]

        return wavefront, psf_center

    def psf(
        self,
        points=[0.0, 0.0, -10000.0],
        ks=PSF_KS,
        wvln=DEFAULT_WAVE,
        spp=SPP_COHERENT,
    ):
        """Single point monochromatic PSF using ray-wave model. The PSF contains all diffraction orders with correct diffraction efficiencies.

        Steps:
            1, Calculate complex wavefield at DOE plane by coherent ray tracing.
            2, Apply DOE phase modulation to the wavefield.
            3, Propagate the wavefield to sensor plane, calculate intensity PSF, crop the valid region and normalize the intensity.

        Args:
            points (torch.Tensor, optional): [x, y, z] coordinates of the point source. Defaults to torch.Tensor([0,0,-10000]).
            ks (int, optional): size of the PSF patch. Defaults to PSF_KS.
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            spp (int, optional): number of rays to sample. Defaults to SPP_COHERENT.

        Returns:
            psf_out (torch.Tensor): PSF patch. Normalized to sum to 1. Shape [ks, ks]
        """
        # Check double precision
        if not torch.get_default_dtype() == torch.float64:
            raise ValueError(
                "Please call HybridLens.double() to set the default dtype to float64 for accurate phase tracing."
            )

        # Check lens last surface
        assert isinstance(self.geolens.surfaces[-1], Phase) or isinstance(
            self.geolens.surfaces[-1], Plane
        ), "The last lens surface should be a DOE."
        geolens, doe = self.geolens, self.doe

        # Compute pupil field by coherent ray tracing
        if isinstance(points, list):
            point0 = torch.tensor(points)
        elif isinstance(points, torch.Tensor):
            point0 = points
        else:
            raise ValueError("point should be a list or a torch.Tensor.")

        wavefront, psfc = self.doe_field(point=point0, wvln=wvln, spp=spp)
        wavefront = wavefront.squeeze(0)  # shape of [H, W]

        # DOE phase modulation. We have to flip the phase map because the wavefront has been flipped
        phase_map = torch.flip(doe.get_phase_map(wvln), [-1, -2])
        wavefront = wavefront * torch.exp(1j * phase_map)

        # Propagate wave field to sensor plane
        h, w = wavefront.shape
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        sensor_field = AngularSpectrumMethod(
            wavefront, z=geolens.d_sensor - doe.d, wvln=wvln, ps=doe.ps, padding=False
        )

        # Compute PSF (intensity distribution)
        psf_inten = sensor_field.abs() ** 2
        psf_inten = (
            F.interpolate(
                psf_inten,
                scale_factor=geolens.sensor_res[0] / h,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # Calculate PSF center index and crop valid PSF region (Consider both interplation and padding)
        if ks is not None:
            h, w = psf_inten.shape[-2:]
            psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
            psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

            # Pad to avoid invalid edge region
            psf_inten_pad = F.pad(
                psf_inten,
                [ks // 2, ks // 2, ks // 2, ks // 2],
                mode="constant",
                value=0,
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            h, w = psf_inten.shape[-2:]
            psf = psf_inten[
                int(h / 2 - h / 4) : int(h / 2 + h / 4),
                int(w / 2 - w / 4) : int(w / 2 + w / 4),
            ]

        # Normalize and convert to float precision
        psf /= psf.sum()  # shape of [ks, ks] or [h, w]
        return diff_float(psf)

    # =====================================================================
    # Visualization
    # =====================================================================
    @torch.no_grad()
    def draw_layout(self, save_name="./DOELens.png", depth=-10000.0, ax=None, fig=None):
        """Draw HybridLens layout with ray-tracing and wave-propagation visualization.

        Shows the lens geometry with ray paths through the refractive elements
        and wave propagation arcs from the DOE to the sensor.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to './DOELens.png'.
            depth (float, optional): Object depth for ray tracing. Defaults to -10000.0.
            ax (matplotlib.axes.Axes, optional): Existing axes to draw on. Defaults to None.
            fig (matplotlib.figure.Figure, optional): Existing figure to use. Defaults to None.

        Returns:
            tuple: (ax, fig) if ax was provided, otherwise saves the figure.
        """
        geolens = self.geolens

        # Draw lens layout
        if ax is None:
            ax, fig = geolens.draw_lens_2d()
            save_fig = True
        else:
            save_fig = False

        # Draw light path
        color_list = ["#CC0000", "#006600", "#0066CC"]
        views = [
            0.0,
            float(np.rad2deg(geolens.rfov) * 0.707),
            float(np.rad2deg(geolens.rfov) * 0.99),
        ]
        arc_radi_list = [0.1, 0.4, 0.7, 1.0, 1.4, 1.8]
        num_rays = 7
        for i, view in enumerate(views):
            # Draw ray tracing
            ray = geolens.sample_point_source_2D(
                depth=depth,
                fov=view,
                num_rays=num_rays,
                entrance_pupil=True,
                wvln=WAVE_RGB[2 - i],
            )
            ray.prop_to(-1.0)

            ray, ray_o_record = geolens.trace(ray=ray, record=True)
            ax, fig = geolens.draw_ray_2d(
                ray_o_record, ax=ax, fig=fig, color=color_list[i]
            )

            # Draw wave propagation
            # Calculate ray center for wave propagation visualization
            ray_center_doe = (
                ((ray.o * ray.is_valid.unsqueeze(-1)).sum(dim=0) / ray.is_valid.sum())
                .cpu()
                .numpy()
            )  # shape [3]
            ray.prop_to(geolens.d_sensor)  # shape [num_rays, 3]
            ray_center_sensor = (
                ((ray.o * ray.is_valid.unsqueeze(-1)).sum(dim=0) / ray.is_valid.sum())
                .cpu()
                .numpy()
            )  # shape [3]

            arc_radi = ray_center_sensor[2] - ray_center_doe[2]
            chief_theta = np.rad2deg(
                np.arctan2(
                    ray_center_sensor[0] - ray_center_doe[0],
                    ray_center_sensor[2] - ray_center_doe[2],
                )
            )
            theta1 = chief_theta - 10
            theta2 = chief_theta + 10

            for j in arc_radi_list:
                arc_radi_j = arc_radi * j
                arc = patches.Arc(
                    (ray_center_sensor[2], ray_center_sensor[0]),
                    arc_radi_j,
                    arc_radi_j,
                    angle=180.0,
                    theta1=theta1,
                    theta2=theta2,
                    color=color_list[i],
                )
                ax.add_patch(arc)

        if save_fig:
            # Save figure
            ax.axis("off")
            ax.set_title("DOE Lens")
            fig.savefig(save_name, bbox_inches="tight", format="png", dpi=600)
            plt.close()
        else:
            return ax, fig

    # =====================================================================
    # Optimization
    # =====================================================================
    def get_optimizer(
        self, doe_lr=1e-4, lens_lr=[1e-4, 1e-4, 1e-2, 1e-5], lr_decay=0.01
    ):
        """Get optimizer for lens and DOE parameters.

        Args:
            doe_lr (float, optional): Learning rate for DOE parameters. Defaults to 1e-4.
            lens_lr (list, optional): Learning rates for lens parameters [d, c, k, a].
                Defaults to [1e-4, 1e-4, 1e-2, 1e-5].
            lr_decay (float, optional): Decay rate for higher-order coefficients. Defaults to 0.01.

        Returns:
            torch.optim.Adam: Configured optimizer for all trainable parameters.
        """
        params = []
        params += self.geolens.get_optimizer_params(lrs=lens_lr, decay=lr_decay)
        params += self.doe.get_optimizer_params(lr=doe_lr)

        optimizer = torch.optim.Adam(params)
        return optimizer

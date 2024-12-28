"""A hybrid refractive-diffractive lens consisting of a geolens and a DOE in the back. Hybrid ray-tracing-wave-propagation is used for differentiable simulation.

This differentiable hybrid lens model can similate:
    1. Aberration of the refractive lens
    2. DOE phase modulation

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu, Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

import json

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .geolens import GeoLens
from .lens import Lens
from .optics.basics import (
    COHERENT_SPP,
    DEFAULT_WAVE,
    WAVE_RGB,
)
from .optics.monte_carlo import forward_integral
from .optics.surfaces import Diffractive_GEO
from .optics.surfaces_diffractive import DOE
from .optics.wave import AngularSpectrumMethod
from .optics.waveoptics_utils import diff_float


class HybridLens(Lens):
    """A hybrid refractive-diffractive lens with a Geolens and a DOE at last.

    This differentiable hybrid lens model can similate:
        1. Aberration of the refractive lens
        2. DOE phase modulation
    """

    def double(self):
        self.geolens.double()
        self.doe.double()

    def read_lens_json(self, lens_path):
        """Read the lens from .json file."""
        # Load geolens
        geolens0 = GeoLens(filename=lens_path)

        with open(lens_path, "r") as f:
            data = json.load(f)

            # Load DOE
            doe_dict = data["DOE"]
            doe0 = DOE(
                l=doe_dict["l"],
                d=doe_dict["d"],
                res=doe_dict["res"],
                fab_ps=doe_dict["fab_ps"],
                param_model=doe_dict["param_model"],
            )
            try:
                doe0.load_doe(doe_dict)
            except Exception:
                print(
                    "When loading DOE, DOE parameter is not found, use random initialization."
                )
                doe0.init_param_model(param_model=doe_dict["param_model"])

            self.doe = doe0

            # Add a DOE surface to GeoLens
            geolens0.surfaces.append(Diffractive_GEO(l=doe0.l, d=doe0.d))
            self.geolens = geolens0

            #
            self.sensor_res = geolens0.sensor_res
            self.pixel_size = geolens0.pixel_size

    def write_lens_json(self, lens_path):
        """Write the lens into .json file."""
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
            surf_dict = {"idx": i + 1}
            surf_dict.update(s.surf_dict())

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
    # Lens operation
    # =====================================================================
    def analysis(self, save_name="./test.png"):
        self.draw_layout(save_name=save_name)

    def prepare_sensor(self, sensor_res):
        self.geolens.prepare_sensor(sensor_res)

        self.sensor_res = self.geolens.sensor_res
        self.pixel_size = self.geolens.pixel_size

    def refocus(self, foc_dist):
        """Refocus the DoeLens to a given depth. Donot move DOE because DOE is installed with geolens in the Siggraph Asia 2024 paper."""
        self.geolens.refocus(foc_dist)

    def draw_layout(self, save_name="./DOELens.png", depth=-10000, ax=None, fig=None):
        """Draw DOELens layout with ray-tracing and wave-propagation."""
        geolens = self.geolens

        # Draw lens layout
        if ax is None:
            ax, fig = geolens.plot_setup2D()
            save_fig = True
        else:
            save_fig = False

        # Draw light path
        color_list = ["#CC0000", "#006600", "#0066CC"]
        views = [0, np.rad2deg(geolens.hfov) * 0.707, np.rad2deg(geolens.hfov) * 0.99]
        arc_radi_list = [0.1, 0.4, 0.7, 1.0, 1.4, 1.8]
        for i, view in enumerate(views):
            # Draw ray tracing
            ray = geolens.sample_point_source_2D(
                depth=depth, view=view, M=5, entrance_pupil=True, wvln=WAVE_RGB[2 - i]
            )
            ray, _, oss = geolens.trace(ray=ray, record=True)
            ax, fig = geolens.plot_raytraces(
                oss, ax=ax, fig=fig, color=color_list[i], ra=ray.ra
            )

            # Draw wave propagation
            ray.prop_to(geolens.d_sensor)
            arc_center = (ray.o[:, 0] * ray.ra).sum() / ray.ra.sum()
            arc_center = arc_center.item()
            # arc_radi = geolens.d_sensor.item() - geolens.surfaces[-1].d.item()
            arc_radi = geolens.d_sensor.item() - self.doe.d.item()
            theta1 = (
                np.rad2deg(
                    np.arctan2(
                        ray.o[0, 0].item() - oss[-1][-1][0],
                        ray.o[0, 2].item() - oss[-1][-1][2],
                    )
                )
                - 4
            )
            theta2 = (
                np.rad2deg(
                    np.arctan2(
                        ray.o[0, 0].item() - oss[0][-1][0],
                        ray.o[0, 2].item() - oss[0][-1][2],
                    )
                )
                + 4
            )

            for j in arc_radi_list:
                arc_radi_j = arc_radi * j
                arc = patches.Arc(
                    (geolens.d_sensor.item(), arc_center),
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

    def get_optimizer(
        self, doe_lr=1e-4, lens_lr=[1e-4, 1e-4, 1e-2, 1e-5], lr_decay=0.01
    ):
        params = []
        params += self.geolens.get_optimizer_params(lr=lens_lr, decay=lr_decay)
        params += self.doe.get_optimizer_params(lr=doe_lr)

        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
        return optimizer

    # =====================================================================
    # PSF-related functions
    # =====================================================================
    def doe_field(self, point, wvln=DEFAULT_WAVE, spp=COHERENT_SPP):
        """Compute the complex wavefront at the DOE plane using coherent ray tracing. This function reimplements geolens.pupil_field() by changing the wavefront computation position to the last surface.

        Args:
            point (torch.Tensor): Tensor of shape (3,) representing the point source position. Defaults to torch.tensor([0.0, 0.0, -10000.0]).
            wvln (float): Wavelength. Defaults to DEFAULT_WAVE.
            spp (int): Samples per pixel. Must be >= 1,000,000 for accurate simulation. Defaults to COHERENT_SPP.

        Returns:

            wavefront: Tensor of shape [H, W] representing the complex wavefront.
            psf_center: List containing the PSF center coordinates [x, y].
        """
        assert spp >= 1_000_000, (
            "Coherent ray tracing spp is too small, "
            "which may lead to inaccurate simulation."
        )
        assert (
            torch.get_default_dtype() == torch.float64
        ), "Default dtype must be set to float64 for accurate phase tracing."

        geolens, doe = self.geolens, self.doe

        if point.dim() == 1:
            point = point.unsqueeze(0)

        # Calculate ray origin in the object space
        scale = geolens.calc_scale_ray(point[:, 2].item())
        point_obj = point.clone()
        point_obj[:, 0] = point[:, 0] * scale * geolens.sensor_size[1] / 2
        point_obj[:, 1] = point[:, 1] * scale * geolens.sensor_size[0] / 2

        # Determine ray center via chief ray
        pointc_chief_ray = geolens.psf_center(point_obj)[0]  # shape [2]

        # Ray tracing
        ray = geolens.sample_from_points(o=point_obj, spp=spp, wvln=wvln)
        ray.coherent = True
        ray, _, _ = geolens.trace(ray)
        ray = ray.prop_to(doe.d)

        # Calculate full-resolution complex field for exit-pupil diffraction
        wavefront = forward_integral(
            ray,
            ps=doe.ps,
            ks=doe.res[0],
            pointc_ref=torch.zeros_like(point[:, :2]),
            coherent=True,
        ).squeeze(
            0
        )  # shape [H, W]

        # Compute PSF center based on chief ray
        psf_center = [
            pointc_chief_ray[0] / geolens.sensor_size[0] * 2,
            pointc_chief_ray[1] / geolens.sensor_size[1] * 2,
        ]

        return wavefront, psf_center

    def psf(
        self,
        points=[0.0, 0.0, -10000.0],
        ks=101,
        wvln=DEFAULT_WAVE,
        spp=COHERENT_SPP,
    ):
        """Single point monochromatic PSF using ray-wave model.

        Steps:
            1, calculate complex wavefield at DOE (pupil) plane by coherent ray tracing.
            2, propagate through DOE to sensor plane, calculate intensity PSF, crop the valid region and normalize.

        Args:
            point (torch.Tensor, optional): [x, y, z] coordinates of the point source. Defaults to torch.Tensor([0,0,-10000]).
            ks (int, optional): size of the PSF patch. Defaults to 101.
            wvln (float, optional): wvln. Defaults to 0.589.
            spp (int, optional): number of rays to sample. Defaults to 1000000.

        Returns:
            psf_out (torch.Tensor): PSF patch. Normalized to sum to 1. Shape [ks, ks]
        """
        # Check double precision
        if not torch.get_default_dtype() == torch.float64:
            raise ValueError(
                "Please call HybridLens.double() to set the default dtype to float64 for accurate phase tracing."
            )

        # Check lens last surface
        assert isinstance(
            self.geolens.surfaces[-1], Diffractive_GEO
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
        psf = diff_float(psf)
        return psf

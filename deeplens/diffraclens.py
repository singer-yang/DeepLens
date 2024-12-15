"""Refractive-diffractive lens with all surface represented using paraxial wave optics."""

import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .optics.basics import DEPTH
from .optics.surfaces_diffractive import DOE, Sensor, ThinLens, Aperture
from .optics.waveoptics_utils import point_source_field
from .lens import Lens


class DoeThinLens(Lens):
    """Paraxial refractive-diffractive lens model. The lens consists of a thin lens, a diffractive optical element (DOE), an aperture, and a sensor. All optical surfaces are modeled using paraxial wave optics.

    This paraxial model is used in most existing computational imaging papers. It is a good approximation for the lens with a small field of view and a small aperture. The model can optimize the DOE but not the thin lens.
    """

    def __init__(self, filename, sensor_res=[1024, 1024]):
        super().__init__(filename, sensor_res)

    def load_file(self, filename):
        """Load lens from a file."""
        self.surfaces = []
        with open(filename, "r") as f:
            data = json.load(f)
            d = 0.0
            for surf_dict in data["surfaces"]:
                if surf_dict["type"] == "Aperture":
                    s = Aperture(r=surf_dict["r"], d=d)
                    self.aperture = s

                elif surf_dict["type"] == "DOE":
                    s = DOE(
                        l=surf_dict["l"],
                        d=d,
                        res=surf_dict["res"],
                        fab_ps=surf_dict["fab_ps"],
                        param_model=surf_dict["param_model"],
                    )
                    if surf_dict["doe_path"] is not None:
                        s.load_doe(surf_dict["doe_path"])
                    self.doe = s

                elif surf_dict["type"] == "ThinLens":
                    s = ThinLens(foclen=surf_dict["foclen"], r=surf_dict["r"], d=d)
                    self.thinlens = s

                elif surf_dict["type"] == "Sensor":
                    s = Sensor(l=surf_dict["l"], d=d, res=surf_dict["res"])
                    self.sensor = s

                else:
                    raise Exception("Surface type not implemented.")

                self.surfaces.append(s)

                if not surf_dict["type"] == "Sensor":
                    d += surf_dict["d_next"]

        self.lens_info = data["info"]

    def write_file(self, filename):
        """Write the lens into a file."""
        # Save DOE to a file
        doe_filename = filename.replace(".json", "_doe.pth")
        self.doe.save_doe(doe_filename)

        # Save lens to a file
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i + 1}

            surf_dict = s.surf_dict()
            if isinstance(s, DOE):
                surf_dict["doe_path"] = doe_filename
            surf_dict.update(surf_dict)

            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = (
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item()
                )

            data["surfaces"].append(surf_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def prop_wave(self, field):
        """Propagate a wavefront through the lens group.

        Args:
            field (Field): Input wavefront.

        Returns:
            field (torch.tensor): Output energy distribution. Shape of [H_sensor, W_sensor]
        """
        for surf in self.surfaces:
            field = surf(field)

        return field

    # =============================================
    # PSF-related functions
    # =============================================
    def psf(self, point=[0, 0, -10000.0], ks=101, wvln=0.589):
        """Calculate monochromatic point PSF by wave propagation approach.

            For the shifted phase issue, refer to "Modeling off-axis diffraction with the least-sampling angular spectrum method".

        Args:
            point (list, optional): Point source position. Defaults to [0, 0, -10000].
            ks (int, optional): PSF kernel size. Defaults to 256.
            wvln (float, optional): wvln. Defaults to 0.55.
            padding (bool, optional): Whether to pad the PSF. Defaults to True.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]
        """
        # Get input field
        x, y, z = point
        sensor = self.sensor
        sensor_l = sensor.l
        field_res = self.doe.res
        scale = -z / sensor.d.item()
        x_obj, y_obj = x * scale * sensor_l / 2, y * scale * sensor_l / 2

        # We have to sample high resolution to meet Nyquist sampling constraint.
        inp_field = point_source_field(
            point=[x_obj, y_obj, z],
            phy_size=[sensor_l, sensor_l],
            res=field_res,
            wvln=wvln,
            fieldz=self.surfaces[0].d.item(),
            device=self.device,
        )

        # Calculate PSF on the sensor. Shape [H_sensor, W_sensor]
        psf_full_res = self.prop_wave(inp_field)[0, 0, :, :]

        # Crop the valid patch of the full-resolution PSF
        coord_c_i = int((1 + y) * sensor.res[0] / 2)
        coord_c_j = int((1 - x) * sensor.res[1] / 2)
        psf_full_res = F.pad(
            psf_full_res, [ks // 2, ks // 2, ks // 2, ks // 2], mode="constant", value=0
        )
        psf_out = psf_full_res[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]

        # Normalize PSF
        psf_out /= psf_out.sum()
        psf_out = torch.flip(psf_out, [0, 1])

        return psf_out

    def draw_psf(self, depth=DEPTH, ks=101, save_name="./psf_doethinlens.png"):
        """Draw on-axis RGB PSF."""
        psf_rgb = self.psf_rgb(point=[0, 0, depth], ks=ks)
        save_image(psf_rgb.unsqueeze(0), save_name, normalize=True)

    # =============================================
    # Utils
    # =============================================
    def get_optimizer(self, lr):
        return self.doe.get_optimizer(lr=lr)

    def draw_layout(self, save_name="./doethinlens.png"):
        """Draw lens setup."""
        fig, ax = plt.subplots()

        # Draw aperture
        d = self.aperture.d.item()
        r = self.aperture.r
        ax.plot([d, d], [r, r + 0.5], "gray")
        ax.plot([d - 0.5, d + 0.5], [r, r], "gray")  # top wedge
        ax.plot([d, d], [-r, -r - 0.5], "gray")
        ax.plot([d - 0.5, d + 0.5], [-r, -r], "gray")  # bottom wedge

        # Draw thinlens
        d = self.thinlens.d.item()
        r = self.thinlens.r
        arrow_length = r
        ax.arrow(
            d,
            -arrow_length,
            0,
            2 * arrow_length,
            head_width=0.5,
            head_length=0.5,
            fc="black",
            ec="black",
            length_includes_head=True,
        )
        ax.arrow(
            d,
            arrow_length,
            0,
            -2 * arrow_length,
            head_width=0.5,
            head_length=0.5,
            fc="black",
            ec="black",
            length_includes_head=True,
        )

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


class DoeLens(Lens):
    """Paraxial diffractive lens model. The lens consists of a diffractive optical element (DOE) and a sensor. DOE is modeled using paraxial wave optics."""

    def __init__(self, filename, sensor_res=[1024, 1024]):
        super().__init__(filename, sensor_res)

    def load_example(self):
        self.doe = DOE(d=0, l=4, res=4000)
        self.doe.init_param_model(param_model="fresnel", f0=50, fresnel_wvln=0.589)
        self.sensor = Sensor(d=50, l=4)
        self.surfaces = [self.doe, self.sensor]

    def load_file(self, filename):
        """Load lens from a file."""
        self.surfaces = []
        with open(filename, "r") as f:
            data = json.load(f)
            d = 0.0
            for surf_dict in data["surfaces"]:
                if surf_dict["type"] == "DOE":
                    s = DOE(
                        l=surf_dict["l"],
                        d=d,
                        res=surf_dict["res"],
                        fab_ps=surf_dict["fab_ps"],
                        param_model=surf_dict["param_model"],
                    )
                    if surf_dict["doe_path"] is not None:
                        s.load_doe(surf_dict["doe_path"])
                    self.doe = s

                elif surf_dict["type"] == "Sensor":
                    s = Sensor(l=surf_dict["l"], d=d, res=surf_dict["res"])
                    self.sensor = s

                else:
                    raise Exception("Surface type not implemented.")

                self.surfaces.append(s)

                if not surf_dict["type"] == "Sensor":
                    d += surf_dict["d_next"]

        self.lens_info = data["info"]

    def write_file(self, filename):
        """Write the lens into a file."""
        # Save DOE to a file
        doe_filename = filename.replace(".json", "_doe.pth")
        self.doe.save_doe(doe_filename)

        # Save lens to a file
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i + 1}

            surf_dict = s.surf_dict()
            if isinstance(s, DOE):
                surf_dict["doe_path"] = doe_filename
            surf_dict.update(surf_dict)

            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = (
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item()
                )

            data["surfaces"].append(surf_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def prop_wave(self, field):
        """Propagate a wavefront through the lens group.

        Args:
            field (Field): Input wavefront.

        Returns:
            field (torch.tensor): Output energy distribution. Shape of [H_sensor, W_sensor]
        """
        for surf in self.surfaces:
            field = surf(field)

        return field

    # =============================================
    # PSF-related functions
    # =============================================
    def psf(self, point=[0, 0, -10000.0], ks=101, wvln=0.589):
        """Calculate monochromatic point PSF by wave propagation approach.

            For the shifted phase issue, refer to "Modeling off-axis diffraction with the least-sampling angular spectrum method".

        Args:
            point (list, optional): Point source position. Defaults to [0, 0, -10000].
            ks (int, optional): PSF kernel size. Defaults to 256.
            wvln (float, optional): wvln. Defaults to 0.55.
            padding (bool, optional): Whether to pad the PSF. Defaults to True.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]
        """
        # Get input field
        x, y, z = point
        sensor = self.sensor
        sensor_l = sensor.l
        field_res = self.doe.res
        scale = -z / sensor.d.item()
        x_obj, y_obj = x * scale * sensor_l / 2, y * scale * sensor_l / 2

        # We have to sample high resolution to meet Nyquist sampling constraint.
        inp_field = point_source_field(
            point=[x_obj, y_obj, z],
            phy_size=[sensor_l, sensor_l],
            res=field_res,
            wvln=wvln,
            fieldz=self.surfaces[0].d.item(),
            device=self.device,
        )

        # Calculate PSF on the sensor. Shape [H_sensor, W_sensor]
        psf_full_res = self.prop_wave(inp_field)[0, 0, :, :]

        # Crop the valid patch of the full-resolution PSF
        coord_c_i = int((1 + y) * sensor.res[0] / 2)
        coord_c_j = int((1 - x) * sensor.res[1] / 2)
        psf_full_res = F.pad(
            psf_full_res, [ks // 2, ks // 2, ks // 2, ks // 2], mode="constant", value=0
        )
        psf_out = psf_full_res[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]

        # Normalize PSF
        psf_out /= psf_out.sum()
        psf_out = torch.flip(psf_out, [0, 1])

        return psf_out

    def draw_psf(
        self,
        depth=DEPTH,
        ks=101,
        save_name="./psf_doelens.png",
        log_scale=True,
        eps=1e-4,
    ):
        """
        Draw on-axis RGB PSF.

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
    # Utils
    # =============================================
    def get_optimizer(self, lr):
        return self.doe.get_optimizer(lr=lr)

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

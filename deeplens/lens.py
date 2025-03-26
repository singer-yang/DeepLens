"""Basic lens class.

When creating a new lens (geolens, diffractivelens, etc.), it is recommended to inherit from the Lens class and re-write core functions.
"""

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from .optics import (
    BLUE_RESPONSE,
    DEPTH,
    GREEN_RESPONSE,
    RED_RESPONSE,
    WAVE_BLUE,
    WAVE_BOARD_BAND,
    WAVE_GREEN,
    WAVE_RED,
    WAVE_RGB,
    DeepObj,
    init_device,
)
from .optics.render_psf import render_psf_map, render_psf


class Lens(DeepObj):
    def __init__(self, filename=None, sensor_res=(1024, 1024), device=None):
        """Initialize a lens class.

        Args:
            filename (str): Path to the lens file.
            sensor_res (tuple, optional): Sensor resolution (W, H). Defaults to (1024, 1024).
            device (str, optional): Device to run the lens. Defaults to None.
        """
        if device is None:
            device = init_device()
        self.device = device

        # Sensor
        self.sensor_res = sensor_res

        # Lens
        if filename is not None:
            self.read_lens(filename)
        self.to(self.device)

    def read_lens(self, filename):
        """Read lens from a file."""
        if filename.endswith(".json"):
            self.read_lens_json(filename)
        else:
            raise Exception("Unknown lens file format.")

    def read_lens_json(self, filename):
        """Read lens from a json file."""
        raise NotImplementedError

    def write_lens(self, filename):
        """Write lens to a file."""
        if filename.endswith(".json"):
            self.write_lens_json(filename)
        else:
            raise Exception("Unknown lens file format.")

    def write_lens_json(self, filename):
        """Write lens to a json file."""
        raise NotImplementedError

    def prepare_sensor(self, sensor_res=[1024, 1024]):
        """Prepare sensor."""
        raise NotImplementedError

    def change_sensor_res(self, sensor_res):
        """Change sensor resolution."""
        if (
            not self.sensor_size[0] * sensor_res[1]
            == self.sensor_size[1] * sensor_res[0]
        ):
            raise Exception("Given sensor resolution does not match sensor size.")
        self.sensor_res = sensor_res
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]

    def post_computation(self):
        """After loading lens, computing some lens parameters."""
        raise NotImplementedError

    # ===========================================
    # PSF-ralated functions
    # ===========================================
    def psf(self, points, ks=51, wvln=0.589, **kwargs):
        """Compute monochrome point PSF. This function is differentiable.

        Args:
            point (tensor): Shape of [N, 3] or [3].
            ks (int, optional): Kernel size. Defaults to 51.
            wvln (float, optional): Wavelength. Defaults to 0.589.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].
        """
        raise NotImplementedError

    def psf_rgb(self, points, ks=51, **kwargs):
        """Compute RGB point PSF.

        Args:
            points (tensor): Shape of [N, 3] or [3].
            ks (int, optional): Kernel size. Defaults to 51.

        Returns:
            psf_rgb: Shape of [3, ks, ks] or [N, 3, ks, ks].
        """
        psfs = []
        for wvln in WAVE_RGB:
            psfs.append(self.psf(points=points, ks=ks, wvln=wvln, **kwargs))
        psf_rgb = torch.stack(psfs, dim=-3)  # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf_rgb

    def psf_narrow_band(self, points, ks=51, **kwargs):
        """Compute RGB PSF considering three wavelengths for each color. Different wavelengths are simpliy averaged for the results, but the sensor response function will be more reasonable.

        Reference:
            https://en.wikipedia.org/wiki/Spectral_sensitivity

        Args:
            points (tensor): Shape of [N, 3] or [3].
            ks (int, optional): Kernel size. Defaults to 51.

        Returns:
            psf: Shape of [3, ks, ks].
        """
        # Red
        psf_r = []
        for _, wvln in enumerate(WAVE_RED):
            psf_r.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_r = torch.stack(psf_r, dim=-3).mean(dim=-3)

        # Green
        psf_g = []
        for _, wvln in enumerate(WAVE_GREEN):
            psf_g.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_g = torch.stack(psf_g, dim=-3).mean(dim=-3)

        # Blue
        psf_b = []
        for _, wvln in enumerate(WAVE_BLUE):
            psf_b.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_b = torch.stack(psf_b, dim=-3).mean(dim=-3)

        # RGB
        psf = torch.stack([psf_r, psf_g, psf_b], dim=-3)
        return psf

    def psf_spectrum(self, points, ks=51, **kwargs):
        """Compute RGB PSF considering full spectrum for each color. A placeholder RGB sensor response function is used to calculate the final PSF. But the actual sensor response function will be more reasonable.

        Reference:
            https://en.wikipedia.org/wiki/Spectral_sensitivity

        Args:
            points (tensor): Shape of [N, 3] or [3].
            ks (int, optional): Kernel size. Defaults to 51.

        Returns:
            psf: Shape of [3, ks, ks].
        """
        # Red
        psf_r = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_r.append(psf * RED_RESPONSE[i])
        psf_r = torch.stack(psf_r, dim=0).sum(dim=0) / sum(RED_RESPONSE)

        # Green
        psf_g = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_g.append(psf * GREEN_RESPONSE[i])
        psf_g = torch.stack(psf_g, dim=0).sum(dim=0) / sum(GREEN_RESPONSE)

        # Blue
        psf_b = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_b.append(psf * BLUE_RESPONSE[i])
        psf_b = torch.stack(psf_b, dim=0).sum(dim=0) / sum(BLUE_RESPONSE)

        # RGB
        psf = torch.stack([psf_r, psf_g, psf_b], dim=0)  # shape [3, ks, ks]
        return psf

    def draw_psf(self, depth=DEPTH, ks=101, save_name="./psf.png"):
        """Draw RGB on-axis PSF."""
        psfs = []
        for wvln in WAVE_RGB:
            psf = self.psf(point=[0, 0, depth], ks=ks, wvln=wvln)
            psfs.append(psf)

        psfs = torch.stack(psfs, dim=0)  # shape [3, ks, ks]
        save_image(psfs.unsqueeze(0), save_name, normalize=True)

    def point_source_grid(
        self, depth, grid=9, normalized=True, quater=False, center=True
    ):
        """Generate point source grid for PSF calculation.

        Args:
            depth (float): Depth of the point source.
            grid (int): Grid size. Defaults to 9, meaning 9x9 grid.
            normalized (bool): Return normalized object source coordinates. Defaults to True, meaning object sources xy coordinates range from [-1, 1].
            quater (bool): Use quater of the sensor plane to save memory. Defaults to False.
            center (bool): Use center of each patch. Defaults to False.

        Returns:
            point_source: Shape of [grid, grid, 3].
        """
        # Compute point source grid
        if grid == 1:
            x, y = torch.tensor([[0.0]]), torch.tensor([[0.0]])
            assert not quater, "Quater should be False when grid is 1."
        else:
            if center:
                # Use center of each patch
                half_bin_size = 1 / 2 / (grid - 1)
                x, y = torch.meshgrid(
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid),
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid),
                    indexing="xy",
                )
            else:
                # Use corner of image sensor
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid),
                    torch.linspace(0.98, -0.98, grid),
                    indexing="xy",
                )

        z = torch.full((grid, grid), depth)
        point_source = torch.stack([x, y, z], dim=-1)

        # Use quater of the sensor plane to save memory
        if quater:
            z = torch.full((grid, grid), depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid // 2 if grid % 2 == 0 else grid // 2 + 1
            bound_j = grid // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        # De-normalize object source coordinates to physical coordinates
        if not normalized:
            scale = self.calc_scale(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    def point_source_radial(self, depth, grid=9, center=False):
        """Compute point radial [0, 1] in the object space to compute PSF grid.

        Args:
            grid (int, optional): Grid size. Defaults to 9.

        Returns:
            point_source: Shape of [grid, 3].
        """
        if grid == 1:
            x = torch.tensor([0.0])
        else:
            # Select center of bin to calculate PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x = torch.linspace(0, 1 - half_bin_size, grid)
            else:
                x = torch.linspace(0, 0.98, grid)

        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source

    def psf_map(self, grid=5, ks=51, depth=DEPTH, wvln=0.589, **kwargs):
        """Compute monochrome PSF map.

        Args:
            grid (int, optional): Grid size. Defaults to 5, meaning 5x5 grid.
            ks (int, optional): Kernel size. Defaults to 51, meaning 51x51 kernel size.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            wvln (float, optional): Wavelength. Defaults to 0.589.

        Returns:
            psf_map: Shape of [1, grid*ks, grid*ks].
        """
        # PSF map grid
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3)

        # Compute PSF map
        psfs = []
        for i in range(points.shape[0]):
            point = points[i, ...]
            psf = self.psf(points=point, ks=ks, wvln=wvln)
            psfs.append(psf)
        psf_map = torch.stack(psfs).unsqueeze(1)

        # Reshape PSF map from [grid*grid, 1, ks, ks] -> [1, grid*ks, grid*ks]
        psf_map = make_grid(psf_map, nrow=grid, padding=0)[0, :, :]

        return psf_map

    def psf_map_rgb(self, grid=5, ks=51, depth=DEPTH, **kwargs):
        """Compute RGB PSF map."""
        psfs = []
        for wvln in WAVE_RGB:
            psf_map = self.psf_map(grid=grid, ks=ks, depth=depth, wvln=wvln, **kwargs)
            psfs.append(psf_map)
        psf_map = torch.stack(psfs, dim=0)  # shape [3, grid*ks, grid*ks]
        return psf_map

    @torch.no_grad()
    def draw_psf_map(
        self, grid=7, ks=101, depth=DEPTH, log_scale=False, save_name="./psf_map.png"
    ):
        """Draw RGB PSF map of the doelens."""
        # Calculate RGB PSF map
        psf_map = self.psf_map_rgb(depth=depth, grid=grid, ks=ks)

        if log_scale:
            # Log scale normalization for better visualization
            psf_map = torch.log(psf_map + 1e-4)  # 1e-4 is an empirical value
            psf_map = (psf_map - psf_map.min()) / (psf_map.max() - psf_map.min())
        else:
            # Linear normalization
            for i in range(0, psf_map.shape[-2], ks):
                for j in range(0, psf_map.shape[-1], ks):
                    local_max = psf_map[:, i : i + ks, j : j + ks].max()
                    if local_max > 0:
                        psf_map[:, i : i + ks, j : j + ks] /= local_max

        fig, ax = plt.subplots(figsize=(10, 10))

        psf_map = psf_map.permute(1, 2, 0).cpu().numpy()
        ax.imshow(psf_map)

        # Add scale bar near bottom-left
        H, W, _ = psf_map.shape
        scale_bar_length = 100
        arrow_length = scale_bar_length / (self.pixel_size * 1e3)
        y_position = H - 20  # a little above the lower edge
        x_start = 20
        x_end = x_start + arrow_length

        ax.annotate(
            "",
            xy=(x_start, y_position),
            xytext=(x_end, y_position),
            arrowprops=dict(arrowstyle="-", color="white"),
            annotation_clip=False,
        )
        ax.text(
            x_end + 5,
            y_position,
            f"{scale_bar_length} μm",
            color="white",
            fontsize=12,
            ha="left",
            va="center",
            clip_on=False,
        )

        # Clean up axes and save
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # ===========================================
    # Image simulation-ralated functions
    # ===========================================
    def render(self, img_obj, depth=DEPTH, method="psf_patch", **kwargs):
        """Differentiable image simulation. This function handles only the differentiable components of image simulation, specifically the optical aberrations. The non-differentiable components (such as noise simulation) are handled separately in the self.render_unprocess() function to ensure more accurate overall image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] Ray tracing-based rendering.
            [3] ...

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. Defaults to "psf".
            **kwargs: Additional arguments for different methods.
        """
        # Check sensor resolution
        if not (
            self.sensor_res[0] == img_obj.shape[-2]
            and self.sensor_res[1] == img_obj.shape[-1]
        ):
            raise Exception("Sensor resolution does not match input image object.")
            H, W = img_obj.shape[-2], img_obj.shape[-1]
            self.prepare_sensor(sensor_res=[H, W])

        # Image simulation (in RAW space)
        if method == "psf_map":
            if "psf_grid" in kwargs and "psf_ks" in kwargs:
                psf_grid, psf_ks = kwargs["psf_grid"], kwargs["psf_ks"]
                img_render = self.render_psf_map(
                    img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
                )
            else:
                img_render = self.render_psf_map(img_obj, depth=depth)

        elif method == "psf_patch":
            if "psf_center" in kwargs and "psf_ks" in kwargs:
                psf_center, psf_ks = kwargs["psf_center"], kwargs["psf_ks"]
                img_render, field_channel = self.render_psf_patch(
                    img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
                )
            else:
                img_render = self.render_psf_patch(img_obj, depth=depth)

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_psf(self, img_obj, depth=DEPTH, psf_center=(0, 0), psf_ks=51):
        """Render image patch using PSF convolution. Better not use this function to avoid confusion."""
        return self.render_psf_patch(
            img_obj, depth=depth, psf_center=psf_center, psf_ks=psf_ks
        )

    def render_psf_patch(self, img_obj, depth=DEPTH, psf_center=(0, 0), psf_ks=51):
        """Render an image patch using PSF convolution, and return positional encoding channel.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [B, C, H, W].
            depth (float): Depth of the object.
            psf_center (tensor): Center of the PSF patch. Shape of [2].
            psf_ks (int): PSF kernel size.

        Returns:
            img_render: Rendered image. Shape of [B, C, H, W].
            field_channel: Positional encoding channel. Shape of [1, H, W].
        """
        # Convert psf_center to tensor
        if isinstance(psf_center, (list, tuple)):
            points = (psf_center[0], psf_center[1], depth)
            points = torch.tensor(points).unsqueeze(0)
        else:
            raise Exception("PSF center must be a list or tuple.")

        # Compute PSF and perform PSF convolution
        psf = self.psf_rgb(points=points, ks=psf_ks).squeeze(0)
        img_render = render_psf(img_obj, psf=psf)

        # Compute positional encoding channel for image patch
        Wobj, Hobj = img_obj.shape[-1], img_obj.shape[-2]
        ps_norm = 2 / self.sensor_res[0]
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                psf_center[0] - Wobj / 2 * ps_norm,
                psf_center[0] + Wobj / 2 * ps_norm,
                Wobj // 2,
                device=self.device,
            ),
            torch.linspace(
                psf_center[1] + Hobj / 2 * ps_norm,
                psf_center[1] - Hobj / 2 * ps_norm,
                Hobj // 2,
                device=self.device,
            ),
            indexing="xy",
        )
        field_channel = torch.sqrt(grid_x**2 + grid_y**2).unsqueeze(0)

        return img_render  # , field_channel

    def render_psf_map(self, img_obj, depth=DEPTH, psf_grid=7, psf_ks=51):
        """Render image using PSF block convolution.

        Note: larger psf_grid and psf_ks are typically better for more accurate rendering, but slower.

        Args:
            img_obj (tensor): Input image object in raw space. Shape of [B, C, H, W].
            depth (float): Depth of the object.
            psf_grid (int): PSF grid size.
            psf_ks (int): PSF kernel size.

        Returns:
            img_render: Rendered image. Shape of [B, C, H, W].
        """
        psf_map = self.psf_map_rgb(grid=psf_grid, ks=psf_ks, depth=depth)
        img_render = render_psf_map(img_obj, psf_map, grid=psf_grid)
        return img_render

    def add_noise(self, img, bit=10):
        """Add sensor read noise and shot noise to RAW space image. Shot and read noise are measured in digital counts (N bit).

        Reference:
            [1] "Unprocessing Images for Learned Raw Denoising."
            [2] https://github.com/timothybrooks/unprocessing

        Args:
            img (tensor): RAW space image. Shape of [N, C, H, W]. Can be either float [0, 1] or integer [0, 2^bit - 1].
            bit (int): Bit depth for noise simulation.

        Returns:
            img: Noisy image. Shape of [N, C, H, W]. Data range [0, 2^bit - 1].
        """
        # Convert float [0,1] to N-bit range if needed
        if img.dtype in [torch.float32, torch.float64] and img.max() <= 1.0:
            img_Nbit = img * (2**bit - 1)
            is_float = True
        else:
            img_Nbit = img
            is_float = False

        # Noise statistics
        if not hasattr(self, "read_noise_std"):
            read_noise_std = 0.0
            print("Read noise standard deviation is not defined.")
        else:
            read_noise_std = self.read_noise_std

        if not hasattr(self, "shot_noise_alpha"):
            shot_noise_alpha = 0.0
            print("Shot noise alpha is not defined.")
        else:
            shot_noise_alpha = self.shot_noise_alpha

        # Add noise to raw image
        noise_std = torch.sqrt(img_Nbit) * shot_noise_alpha + read_noise_std
        noise = torch.randn_like(img_Nbit) * noise_std
        img_Nbit = img_Nbit + noise
        img_Nbit = torch.clamp(img_Nbit, 0, 2**bit - 1)

        # Convert N-bit image to float [0, 1]
        if is_float:
            img = img_Nbit / (2**bit - 1)
        else:
            img = img_Nbit
        return img

    # ===========================================
    # Visualization-ralated functions
    # ===========================================
    def draw_layout(self):
        """Draw lens layout."""
        raise NotImplementedError

    # ===========================================
    # Optimization-ralated functions
    # ===========================================
    def activate_grad(self, activate=True):
        """Activate gradient for each surface."""
        raise NotImplementedError

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-3]):
        """Get optimizer parameters for different lens parameters."""
        raise NotImplementedError

    def get_optimizer(self, lr=[1e-4, 1e-4, 0, 1e-3]):
        """Get optimizer."""
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)
        return optimizer

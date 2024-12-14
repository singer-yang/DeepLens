"""Basic lens class.

When creating a new lens class, it is recommended to inherit from the Lens class and re-write core functions.
"""

import torch
from .optics import (
    DeepObj,
    init_device,
    WAVE_RGB,
    WAVE_BLUE,
    WAVE_GREEN,
    WAVE_RED,
    WAVE_BOARD_BAND,
    RED_RESPONSE,
    GREEN_RESPONSE,
    BLUE_RESPONSE,
    DEPTH,
    render_psf_map,
)


class Lens(DeepObj):
    """Geolens class. A geometric lens consisting of refractive surfaces, simulate with ray tracing. May contain diffractive surfaces, but still use ray tracing to simulate."""

    def __init__(self, filename=None, sensor_res=[1024, 1024]):
        """A lens class."""
        super(Lens, self).__init__()
        self.device = init_device()

        # Load lens file
        self.lens_name = filename
        self.load_file(filename, sensor_res)
        self.to(self.device)

        # Lens calculation
        self.prepare_sensor(sensor_res)
        self.post_computation()

    def load_file(self, filename):
        """Load lens from a file."""
        raise NotImplementedError

    def write_file(self, filename):
        """Write lens to a file."""
        raise NotImplementedError

    def prepare_sensor(self, sensor_res=[1024, 1024]):
        """Prepare sensor."""
        raise NotImplementedError

    def post_computation(self):
        """After loading lens, computing some lens parameters."""
        raise NotImplementedError

    # ===========================================
    # PSF-ralated functions
    # ===========================================
    def psf(self, points, ks=51, wvln=0.589, **kwargs):
        """Compute monochrome point PSF. This function is differentiable."""
        pass

    def psf_rgb(self, points, ks=51, **kwargs):
        """Compute RGB point PSF. This function is differentiable."""
        psfs = []
        for wvln in WAVE_RGB:
            psfs.append(self.psf(points=points, ks=ks, wvln=wvln, **kwargs))
        psf_rgb = torch.stack(psfs, dim=-3)  # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf_rgb

    def psf_narrow_band(self, points, ks=51, **kwargs):
        """Should be migrated to psf_rgb."""
        psf_r = []
        for i, wvln in enumerate(WAVE_RED):
            psf_r.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_r = torch.stack(psf_r, dim=-3).mean(dim=-3)

        psf_g = []
        for i, wvln in enumerate(WAVE_GREEN):
            psf_g.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_g = torch.stack(psf_g, dim=-3).mean(dim=-3)

        psf_b = []
        for i, wvln in enumerate(WAVE_BLUE):
            psf_b.append(self.psf(points=points, wvln=wvln, ks=ks, **kwargs))
        psf_b = torch.stack(psf_b, dim=-3).mean(dim=-3)

        psf = torch.stack([psf_r, psf_g, psf_b], dim=-3)
        return psf

    def psf_spectrum(self, points, ks=51, **kwargs):
        """Should be migrated to psf_rgb."""
        psf_r = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_r.append(psf * RED_RESPONSE[i])
        psf_r = torch.stack(psf_r, dim=0).sum(dim=0) / sum(RED_RESPONSE)

        psf_g = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_g.append(psf * GREEN_RESPONSE[i])
        psf_g = torch.stack(psf_g, dim=0).sum(dim=0) / sum(GREEN_RESPONSE)

        psf_b = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(points=points, ks=ks, wvln=wvln, **kwargs)
            psf_b.append(psf * BLUE_RESPONSE[i])
        psf_b = torch.stack(psf_b, dim=0).sum(dim=0) / sum(BLUE_RESPONSE)

        psf = torch.stack([psf_r, psf_g, psf_b], dim=0)  # shape [3, ks, ks]
        return psf

    def psf_map(self, grid=21, ks=51, depth=-20000.0, **kwargs):
        """Compute PSF map."""
        pass

    # ===========================================
    # Rendering-ralated functions
    # ===========================================
    def render(self, img, method="psf", noise_std=0.01):
        """PSF based rendering or image based rendering."""
        if not (
            self.sensor_res[0] == img.shape[-2] and self.sensor_res[1] == img.shape[-1]
        ):
            self.prepare_sensor(sensor_res=[img.shape[-2], img.shape[-1]])

        if method == "psf":
            # Note: larger psf_grid and psf_ks are better
            psf_map = self.psf_map(grid=psf_grid, ks=psf_ks, depth=depth)
            img_render = render_psf_map(img, psf_map, grid=psf_grid)
        else:
            raise Exception("Unknown method.")

        # Add sensor noise
        if noise_std > 0:
            img_render = img_render + torch.randn_like(img_render) * noise_std

        return img_render

    def add_noise(self, img, read_noise_std=0.01, shot_noise_alpha=1.0):
        """Add both read noise and shot noise. Note: for an accurate noise model, we need to convert back to RAW space."""
        noise_std = torch.sqrt(img) * shot_noise_alpha + read_noise_std
        noise = torch.randn_like(img) * noise_std
        img = img + noise
        return img

    # ===========================================
    # Visualization-ralated functions
    # ===========================================
    def draw_layout(self):
        """Draw lens layout."""
        raise NotImplementedError

    def draw_psf_map(
        self,
        grid=7,
        depth=DEPTH,
        ks=101,
        log_scale=False,
        recenter=True,
        save_name="./psf",
    ):
        """Draw lens RGB PSF map."""
        raise NotImplementedError

    def draw_render_image(self, image):
        """Draw input and simulated images."""
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

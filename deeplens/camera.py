"""Camera class works as a renderer in an end-to-end pipeline. It contains a lens and a sensor."""

import torch

from deeplens import GeoLens
from deeplens.sensor import RGBSensor


class Renderer:
    """In the future Renderer will be replaced as Camera to be integrated into DeepLens code"""

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def set_device(self, device):
        """Set the device for rendering."""
        self.device = device

    def move_to_device(self, data_dict):
        """Move data to device."""
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device)
        return data_dict

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Camera(Renderer):
    def __init__(
        self,
        lens_file,
        sensor_size=(5, 5),
        sensor_res=(1024, 1024),
        device=None,
    ):
        super().__init__(device)

        # Lens
        self.lens = GeoLens(lens_file, device=device)
        self.lens.set_sensor(sensor_res=sensor_res, sensor_size=sensor_size)

        # Sensor
        self.sensor = RGBSensor(
            res=sensor_res,
            bit=10,
            black_level=64,
            iso_base=100,
            read_noise_std=0.5,
            shot_noise_std_alpha=0.4,
        ).to(device)

    def __call__(self, data_dict):
        """Render a blurry and noisy RGB image considering camera and lens."""
        return self.render(data_dict)

    def render(self, data_dict):
        """Render a blurry and noisy RGB image considering camera and lens.

        Args:
            data_dict (dict): Dictionary containing essential information for image simulation, for example:
                {
                    "img": rgb image (torch.Tensor), (B, 3, H, W), [0, 1]
                    "iso": iso (int),
                    "output_type": "rggbi",
                }
        """
        data_dict = self.move_to_device(data_dict)
        img = data_dict["img"]
        iso = data_dict["iso"]
        output_type = data_dict["output_type"]

        # Unprocess to raw space
        sensor = self.sensor
        img_raw = sensor.unprocess(img)  # (B, 3, H, W), [0, 1]

        # Lens simulation in raw space
        img_blur = self.render_lens(img_raw)  # (B, 3, H, W), [0, 1]

        # Convert to bayer space
        bayer_gt = sensor.raw2bayer(img_raw)  # (B, 1, H, W), [0, 2**bit - 1]
        bayer_blur = sensor.raw2bayer(img_blur)  # (B, 1, H, W), [0, 2**bit - 1]

        # Simulate sensor noise
        bayer_blur_noise = sensor.simu_noise(
            bayer_blur, iso
        )  # (B, 1, H, W), [~black_level, 2**bit - 1]

        # Pack output for network training
        img_rgb = self.pack_output(
            bayer_gt=bayer_gt,
            bayer_blur_noise=bayer_blur_noise,
            iso=iso,
            output_type=output_type[0],
        )
        return img_rgb

    def render_lens(self, img_raw, field=None):
        """Render a blurry rgb raw image with the lens.

        Here we adopt the full sensor resolution rendering, while a PSF-based patch rendering is also possible.
        """
        return self.lens.render(img_raw)

    def pack_output(
        self,
        bayer_gt,
        bayer_blur_noise,
        iso,
        iso_scale=1000,
        field_center=((0.0, 0.0)),
        output_type="rggbi",
    ):
        """Pack Prepare network input for training and testing.

        Args:
            bayer_gt (torch.Tensor): Bayer image (B, 1, H, W), [~black_level, 2**bit - 1]
            bayer_blur_noise (torch.Tensor): Bayer image with noise (B, 1, H, W), [~black_level, 2**bit - 1]
            iso (torch.Tensor): ISO value (B,)
            iso_scale (int): ISO scale
            field_center (tuple): Center of the field of view (B, 2)
            output_type (str): Output type

        Returns:
            rggbi_blur_noise (torch.Tensor): RGGB image with noise (B, C, H, W)
            rggbi_gt (torch.Tensor): RGGB image (B, C, H, W)
        """
        sensor = self.sensor

        # Prepare network input
        if output_type == "rgb":
            rgb_gt = sensor.isp(bayer_gt)
            rgb_blur_noise = sensor.isp(bayer_blur_noise)
            return rgb_blur_noise, rgb_gt

        elif output_type == "rggbi":
            H, W = bayer_blur_noise.shape[-2:]

            # RGGB channels
            rggb_gt = sensor.bayer2rggb(bayer_gt)  # (B, 4, H, W), [0, 1]
            rggb_blur_noise = sensor.bayer2rggb(
                bayer_blur_noise
            )  # (B, 4, H, W), [0, 1]

            # ISO channel (B, 1, H, W)
            iso_channel = (
                torch.ones_like(rggb_gt[:, 0, :, :].unsqueeze(1))
                * iso.view(-1, 1, 1, 1)
                / iso_scale
            )

            rggbi_blur_noise = torch.cat([rggb_blur_noise, iso_channel], dim=1)
            return rggbi_blur_noise, rggb_gt

        elif output_type == "rggbif":
            H, W = bayer_blur_noise.shape[-2:]

            # RGGB channels
            rggb_gt = sensor.bayer2rggb(bayer_gt)  # (B, 4, H, W), [0, 1]
            rggb_blur_noise = sensor.bayer2rggb(
                bayer_blur_noise
            )  # (B, 4, H, W), [0, 1]

            # ISO channel (B, 1, H, W)
            iso_channel = (
                torch.ones_like(rggb_gt[:, 0, :, :].unsqueeze(1))
                * iso.view(-1, 1, 1, 1)
                / iso_scale
            )

            # Field channel (B, 1, H, W)
            B = bayer_blur_noise.shape[0]
            field_channels = []

            for b in range(B):
                grid_x, grid_y = torch.meshgrid(
                    torch.linspace(
                        field_center[b, 0] - W / 2 * self.pixel_size,
                        field_center[b, 0] + W / 2 * self.pixel_size,
                        W // 2,
                        device=bayer_blur_noise.device,
                    ),
                    torch.linspace(
                        field_center[b, 1] + H / 2 * self.pixel_size,
                        field_center[b, 1] - H / 2 * self.pixel_size,
                        H // 2,
                        device=bayer_blur_noise.device,
                    ),
                    indexing="xy",
                )
                field_channel = torch.sqrt(grid_x**2 + grid_y**2).unsqueeze(0)
                field_channels.append(field_channel)

            field_channel = torch.cat(field_channels, dim=0).unsqueeze(1)

            # Concatenate all channels
            rggbif_blur_noise = torch.cat(
                [rggb_blur_noise, iso_channel, field_channel], dim=1
            )
            return rggbif_blur_noise, rggb_gt

        else:
            raise NotImplementedError(f"Invalid output type: {output_type}")

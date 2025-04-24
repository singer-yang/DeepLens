import torch

from .sensor import Sensor
from .isp import InvertibleISP

class RGBSensor(Sensor):
    """RGB sensor. Unprocess and process in RAW RGB space."""

    def __init__(
        self,
        bit=10,
        black_level=64,
        res=(4000, 3000),
        size=(8.0, 6.0),
        bayer_pattern="rggb",
        iso_base=100,
        read_noise_std=0.5,
        shot_noise_std_alpha=0.4,
        shot_noise_std_beta=0.0,
        wavelengths=None,
        red_response=None,
        green_response=None,
        blue_response=None,
    ):
        super().__init__(
            bit=bit,
            black_level=black_level,
            res=res,
            size=size,
            iso_base=iso_base,
            read_noise_std=read_noise_std,
            shot_noise_std_alpha=shot_noise_std_alpha,
            shot_noise_std_beta=shot_noise_std_beta,
        )

        # Initialize ISP
        self.isp = InvertibleISP(
            bit=bit,
            black_level=black_level,
            bayer_pattern=bayer_pattern,
        )

        # Initialize spectral response curves
        self.wavelengths = wavelengths
        self.red_response = red_response
        self.green_response = green_response
        self.blue_response = blue_response
        if self.wavelengths is not None:
            self.red_response = torch.tensor(red_response) / sum(green_response)
            self.green_response = torch.tensor(green_response) / sum(green_response)
            self.blue_response = torch.tensor(blue_response) / sum(green_response)

    @classmethod
    def from_config(cls, config):
        """Create a sensor from a config file or dictionary.

        Args:
            config_path: Path to the JSON config file (optional if config is provided)
            config: Configuration dictionary (optional if config_path is provided)

        Returns:
            An instance of RGBSensor initialized with the config parameters
        """
        # Extract parameters with defaults
        res = config.get("res", (4000, 3000))
        size = config.get("size", (8.0, 6.0))
        bit = config.get("bit", 10)
        black_level = config.get("black_level", 64)
        iso_base = config.get("iso_base", 100)
        read_noise_std = config.get("read_noise_std", 0.5)
        shot_noise_std_alpha = config.get("shot_noise_std_alpha", 0.4)
        bayer_pattern = config.get("bayer_pattern", "rggb")

        # Get spectral response curves
        wavelengths = config.get("wavelengths", None)
        red_response = config.get("red_response", None)
        green_response = config.get("green_response", None)
        blue_response = config.get("blue_response", None)

        # Create and return a new sensor instance
        return cls(
            res=res,
            size=size,
            bit=bit,
            black_level=black_level,
            iso_base=iso_base,
            read_noise_std=read_noise_std,
            shot_noise_std_alpha=shot_noise_std_alpha,
            bayer_pattern=bayer_pattern,
            wavelengths=wavelengths,
            red_response=red_response,
            green_response=green_response,
            blue_response=blue_response,
        )
    
    def to(self, device):
        super().to(device)
        if self.wavelengths is not None:
            self.red_response = self.red_response.to(device)
            self.green_response = self.green_response.to(device)
            self.blue_response = self.blue_response.to(device)
        return self

    def response_curve(self, img_spectral):
        """Apply response curve to the spectral image to get the raw image.

        Args:
            img_spectral: Spectral image

        Returns:
            img_raw: Raw image
        """
        if self.wavelengths is not None:
            img_raw = torch.zeros(
                (
                    img_spectral.shape[0],
                    3,
                    img_spectral.shape[2],
                    img_spectral.shape[3],
                ),
                device=img_spectral.device,
            )
            img_raw[:, 0, :, :] = (img_spectral * self.red_response.view(1, -1, 1, 1)).sum(dim=1)
            img_raw[:, 1, :, :] = (img_spectral * self.green_response.view(1, -1, 1, 1)).sum(dim=1)
            img_raw[:, 2, :, :] = (img_spectral * self.blue_response.view(1, -1, 1, 1)).sum(dim=1)
        else:
            assert img_spectral.shape[1] == 3, (
                "No spectral response curves provided, input image must have 3 channels"
            )
            img_raw = img_spectral

        return img_raw

    def forward(self, img_nbit, iso):
        """Simulate sensor output with noise and ISP.

        Args:
            img_nbit: Tensor of shape (B, 3, H, W), range [~black_level, 2**bit - 1]
            iso: ISO value as int

        Returns:
            img_noise: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        img_raw = self.response_curve(img_nbit)
        img_noise = self.simu_noise(img_raw, iso)
        img_noise = self.isp(img_noise)
        return img_noise

    # ===============================
    # Unprocess
    # ===============================
    def unprocess(self, image, in_type="rgb"):
        """Unprocess an image to unbalanced RAW RGB space.

        Args:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
            in_type: Input image type, either "rgb" or "linear_rgb"

        Returns:
            image: Tensor of shape (B, 3, H, W), range [0, 1] in raw space
        """
        isp = self.isp

        # Inverse gamma correction
        if in_type == "linear_rgb":
            pass
        elif in_type == "rgb":
            image = isp.gamma.reverse(image)
        else:
            raise ValueError(f"Invalid input type: {in_type}")

        # Inverse color correction matrix
        image = isp.ccm.reverse(image)

        # Inverse auto white balance
        image = isp.awb.reverse(image)  # (B, 3, H, W), [0, 1]

        return image

    def raw2bayer(self, img_raw):
        """Unprocess the raw image from [0, 1] to [~black_level, 2**bit - 1].

        Args:
            img_raw: Tensor of shape (B, 3, H, W), range [0, 1]

        Returns:
            bayer_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]
        """
        bayer_float = self.isp.demosaic.reverse(img_raw)
        bayer_nbit = (
            bayer_float * (2**self.bit - 1 - self.black_level) + self.black_level
        )
        bayer_nbit = torch.round(bayer_nbit)
        return bayer_nbit

    def sample_augmentation(self):
        """Enable augmentation for ISP modules."""
        self.isp.gamma.sample_augmentation()
        self.isp.ccm.sample_augmentation()
        self.isp.awb.sample_augmentation()

    def reset_augmentation(self):
        """Reset augmentation for ISP modules."""
        self.isp.gamma.reset_augmentation()
        self.isp.ccm.reset_augmentation()
        self.isp.awb.reset_augmentation()

    # ===============================
    # Packing and unpacking
    # ===============================
    def process2rgb(self, image, in_type=None):
        """Process an image to a RGB image.

        Args:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
            in_type: Input image type, either "rgb" or "bayer" or "rggb"

        Returns:
            image: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        # Determine input type
        if in_type is None and image.shape[1] == 1:
            in_type = "bayer"
        elif in_type is None and image.shape[1] == 3:
            in_type = "rgb"
        elif in_type is None and image.shape[1] == 4:
            in_type = "rggb"
        else:
            raise ValueError(f"Invalid input type: {in_type}")

        # Process to RGB
        if in_type == "rgb":
            image = self.isp(image)
        elif in_type == "rggb":
            bayer = self.rggb2bayer(image)
            image = self.isp(bayer)
        elif in_type == "bayer":
            bayer = image
            image = self.isp(bayer)
        else:
            raise ValueError(f"Invalid input type: {in_type}")

        return image

    def bayer2rggb(self, bayer_nbit):
        """Convert RAW bayer image to RAW RGB image.

        Args:
            bayer_nbit: Tensor of shape (B, 1, H, W), range [~black_level, 2**bit - 1]

        Returns:
            rggb: Tensor of shape (B, 3, H, W), range [0, 1]
        """
        black_level = self.black_level
        bit = self.bit

        if len(bayer_nbit.shape) == 2:
            bayer_nbit = bayer_nbit.unsqueeze(0).unsqueeze(0)
            single_image = True
        else:
            single_image = False

        B, _, H, W = bayer_nbit.shape
        bayer_rggb = torch.zeros(
            (B, 4, H // 2, W // 2), dtype=bayer_nbit.dtype, device=bayer_nbit.device
        )

        bayer_rggb[:, 0, :, :] = bayer_nbit[:, 0, 0:H:2, 0:W:2]
        bayer_rggb[:, 1, :, :] = bayer_nbit[:, 0, 0:H:2, 1:W:2]
        bayer_rggb[:, 2, :, :] = bayer_nbit[:, 0, 1:H:2, 0:W:2]
        bayer_rggb[:, 3, :, :] = bayer_nbit[:, 0, 1:H:2, 1:W:2]

        # Data range [black_level, 2**bit - 1] -> [0, 1]
        rggb = (bayer_rggb - black_level) / (2**bit - 1 - black_level)

        if single_image:
            rggb = rggb.squeeze(0)

        return rggb

    def rggb2bayer(self, rggb):
        """Convert RGGB image to RAW Bayer.

        Args:
            rggb: Tensor of shape [4, H/2, W/2] or [B, 4, H/2, W/2], range [0, 1]

        Returns:
            bayer: Tensor of shape [1, H, W] or [B, 1, H, W], range [~black_level, 2**bit - 1]
        """
        black_level = self.black_level
        bit = self.bit

        if len(rggb.shape) == 3:
            rggb = rggb.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        B, _, H, W = rggb.shape
        bayer = torch.zeros((B, 1, H * 2, W * 2), dtype=rggb.dtype).to(rggb.device)

        bayer[:, 0, 0 : 2 * H : 2, 0 : 2 * W : 2] = rggb[:, 0, :, :]
        bayer[:, 0, 0 : 2 * H : 2, 1 : 2 * W : 2] = rggb[:, 1, :, :]
        bayer[:, 0, 1 : 2 * H : 2, 0 : 2 * W : 2] = rggb[:, 2, :, :]
        bayer[:, 0, 1 : 2 * H : 2, 1 : 2 * W : 2] = rggb[:, 3, :, :]

        # Data range [0, 1] -> [0, 2**bit-1]
        bayer = torch.round(bayer * (2**bit - 1 - black_level) + black_level)

        if single_image:
            bayer = bayer.squeeze(0)

        return bayer


"""Differentiable Image Signal Processing (ISP)."""

import numpy as np
import torch
import torch.nn.functional as F


class ISP:
    """Image Signal Processing (ISP) pipeline to generate RGB images from RAW images.

    Rerference:
        [1] https://github.com/QiuJueqin/fast-openISP/tree/master
        [2] https://github.com/timothybrooks/hdr-plus/tree/master
    """

    def __init__(self, bit=10, black_level=64):
        self.bit = bit
        self.black_level = black_level

        self.dead_pixel_threshold = 30

        self.demosaic_method = "naive"

        self.awb_method = "naive"
        self.awb_gains = [2.0, 1.0, 1.8]

        # Reference data from https://github.com/QiuJueqin/fast-openISP/blob/master/configs/nikon_d3200.yaml#L57
        # Alternative data from https://github.com/timothybrooks/hdr-plus/blob/master/src/finish.cpp#L626
        self.ccm_matrix = torch.tensor(
            [
                [1.8506, -0.7920, -0.0605],
                [-0.1562, 1.6455, -0.4912],
                [0.0176, -0.5439, 1.5254],
                [0.0, 0.0, 0.0],
            ]
        )

        self.tone_mapping_param = 1.0

        self.gamma_param = 2.2

    def __call__(self, *args, **kwds):
        return self.diff_isp(*args, **kwds)

    def diff_isp(
        self,
        bayer_Nbit,
    ):
        """A basic differentiable and invertible ISP pipeline.

        Rerference:
            [1] Architectural Analysis of a Baseline ISP Pipeline. https://link.springer.com/chapter/10.1007/978-94-017-9987-4_2. (page 23, 50)

        Args:
            bayer_Nbit: Input tensor of shape [B, 1, H, W], data range [64, 1023].

        Returns:
            rgb: Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        # 0. Sensor response curve
        # Light illuminance is converted to raw bayer image with sensor response curve.

        # 1. Black level subtraction
        bayer = self.blc(bayer_Nbit, bit=self.bit, black_level=self.black_level)

        # Noise reduction

        # 2. White balance
        bayer = self.awb(bayer, awb_method=self.awb_method)

        # 3. Demosaic
        rgb = self.demosaic(bayer, method=self.demosaic_method)

        # Lens correction

        # 4. Color correction matrix
        rgb = self.ccm(rgb, ccm_matrix=self.ccm_matrix)

        # 5. Gamma correction (from N-bit to 8-bit)
        rgb = self.gamma_correction(rgb, gamma_param=self.gamma_param, quantize=True)

        # 6. Color space conversion
        # rgb = self.csc(rgb)

        return rgb

    def isp_linearRGB(
        self,
        rgb,
    ):
        """Simplified ISP pipeline for unprocess and image simulation.

        Args:
            rgb (tensor): Linear RGB image only with blc and demosaic, shape [B, 3, H, W], data range [0, 1].

        Returns:
            rgb (tensor): Output RGB image, shape [B, 3, H, W], data range [0, 1].
        """
        # 1. White balance
        rgb = self.awb(rgb, awb_method=self.awb_method)

        # 2. Color correction matrix
        rgb = self.ccm(rgb, ccm_matrix=self.ccm_matrix)

        # 3. Gamma correction
        rgb = self.gamma_correction(rgb, gamma_param=self.gamma_param, quantize=True)

        return rgb

    def hdr_plus(
        self,
        bayer_Nbit,
    ):
        """HDR+ burst ISP pipeline by Google, here we only use single image ISP part.

        Reference:
            [1] https://www.timothybrooks.com/tech/hdr-plus/
            [2] https://github.com/timothybrooks/hdr-plus/blob/master/src/finish.cpp#L767

        Args:
            bayer_Nbit: Input tensor of shape [B, 1, H, W], data range [64, 1023].

        Returns:
            rgb: Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        bit = self.bit

        # 1. Black level subtraction
        bayer = self.blc(bayer_Nbit, bit=bit, black_level=self.black_level)

        # 2. White balance
        bayer = self.awb(bayer, awb_method=self.awb_method)

        # 3. Demosaic
        rgb = self.demosaic(bayer, demosaic_method=self.demosaic_method)

        # 4. Chroma denoise

        # 5. sRGB color correction

        # 6. Tone mapping

        # 7. Gamma correction
        rgb = self.gamma_correction(rgb, gamma_param=self.gamma_param)

        # 8. Global contrast

        # 9. Sharpening

        return rgb

    # ====================================================================
    # ISP functions
    # ====================================================================
    @staticmethod
    def anti_aliasing_filter(bayer):
        """Anti-Aliasing Filter (AAF)

        Args:
            bayer: Input tensor of shape [B, C, H, W], data range [0, 1]

        Returns:
            Filtered bayer tensor of same shape as input
        """
        # Convert to uint32 for calculations
        bayer = bayer.to(torch.int32)

        # Pad the input with reflection padding
        padded = F.pad(bayer, (2, 2, 2, 2), mode="reflect")

        # Get all 9 shifted versions for 3x3 window
        shifts = []
        for i in range(3):
            for j in range(3):
                shifts.append(
                    padded[:, :, i : i + bayer.shape[2], j : j + bayer.shape[3]]
                )

        # Initialize result tensor
        result = torch.zeros_like(shifts[0], dtype=torch.int32)

        # Apply weights: center pixel (index 4) gets weight 8, others get weight 1
        for i, shifted in enumerate(shifts):
            weight = 8 if i == 4 else 1
            result += weight * shifted

        # Right shift by 4 (divide by 16)
        result = result >> 4

        return result.to(torch.uint16)

    @staticmethod
    def awb(bayer, awb_method="naive", bayer_pattern="RGGB"):
        """
        Auto White Balance (AWB) implementation in PyTorch

        Args:
            bayer: Input tensor of shape [B, C, H, W], data range [0, 1]
            r_gain: Red gain value
            gr_gain: Green-Red gain value
            gb_gain: Green-Blue gain value
            b_gain: Blue gain value
            bayer_pattern: Bayer pattern arrangement (default: 'RGGB')

        Returns:
            White balanced bayer tensor of same shape as input
        """
        # Gain values for different methods
        if awb_method == "naive":
            r_gain, gr_gain, gb_gain, b_gain = 2.0, 1.0, 1.0, 1.8
        elif awb_method == "auto":
            # Compute the average of each channel
            r_avg = bayer[:, 0, :, :].mean().item()
            g_avg = bayer[:, 1, :, :].mean().item()
            b_avg = bayer[:, 2, :, :].mean().item()

            r_gain = g_avg / r_avg if r_avg > 0 else 1.0
            gr_gain = 1.0
            gb_gain = 1.0
            b_gain = g_avg / b_avg if b_avg > 0 else 1.0
        else:
            raise ValueError("Invalid awb_method value")

        # Create gain mask according to Bayer pattern
        # H, W = bayer.shape[2], bayer.shape[3]
        gain_mask = torch.zeros_like(bayer)

        if bayer_pattern == "RGGB":
            gain_mask[:, :, 0::2, 0::2] = r_gain
            gain_mask[:, :, 0::2, 1::2] = gr_gain
            gain_mask[:, :, 1::2, 0::2] = gb_gain
            gain_mask[:, :, 1::2, 1::2] = b_gain
        elif bayer_pattern == "BGGR":
            gain_mask[:, :, 0::2, 0::2] = b_gain
            gain_mask[:, :, 0::2, 1::2] = gb_gain
            gain_mask[:, :, 1::2, 0::2] = gr_gain
            gain_mask[:, :, 1::2, 1::2] = r_gain
        elif bayer_pattern == "GRBG":
            gain_mask[:, :, 0::2, 0::2] = gr_gain
            gain_mask[:, :, 0::2, 1::2] = r_gain
            gain_mask[:, :, 1::2, 0::2] = b_gain
            gain_mask[:, :, 1::2, 1::2] = gb_gain
        elif bayer_pattern == "GBRG":
            gain_mask[:, :, 0::2, 0::2] = gb_gain
            gain_mask[:, :, 0::2, 1::2] = b_gain
            gain_mask[:, :, 1::2, 0::2] = r_gain
            gain_mask[:, :, 1::2, 1::2] = gr_gain

        # Apply gains and shift right by 10 (divide by 1024)
        result = bayer * gain_mask

        # Clip values
        result = torch.clamp(result, 0, 1.0)

        return result

    @staticmethod
    def blc(image, bit=10, black_level=64):
        """Black level correction.

        Args:
            image (tensor): Image, int data range [0, 1023].
            black_level (int): Black level, default 64.

        Returns:
            image (tensor): Image, float data range [0, 1].
        """
        if isinstance(image, np.ndarray):
            image_float = (image.astype(np.float32) - black_level) / (
                2**bit - black_level
            )
            image_float = np.clip(image_float, 0, 1)
        elif isinstance(image, torch.Tensor):
            image_float = (image.float() - black_level) / (2**bit - black_level)
            image_float = torch.clip(image_float, 0, 1)
        else:
            raise ValueError("Invalid image type")

        return image_float

    @staticmethod
    def ccm(rgb_image, ccm_matrix):
        """Color correction matrix. Convert RGB image to sensor color space.

        Args:
            rgb_image: Input tensor of shape [B, 3, H, W] in RGB format.

        Returns:
            rgb_corrected: Corrected RGB image in sensor color space.
        """
        # Extract matrix and bias
        matrix = ccm_matrix[:3, :].float()  # Shape: (3, 3)
        bias = ccm_matrix[3, :].float().view(1, 3, 1, 1)  # Shape: (1, 3, 1, 1)

        # Apply CCM
        # Reshape rgb_image to [B, H, W, 3] for matrix multiplication
        rgb_image_perm = rgb_image.permute(0, 2, 3, 1)  # [B, H, W, 3]
        rgb_corrected = torch.matmul(rgb_image_perm, matrix.T) + bias.squeeze()
        rgb_corrected = rgb_corrected.permute(0, 3, 1, 2)  # [B, 3, H, W]

        return rgb_corrected

    @staticmethod
    def dead_pixel_correction(bayer, diff_threshold=30):
        """Dead pixel correction implementation.

        Args:
            bayer: Input tensor of shape [B, C, H, W]
            diff_threshold: Threshold for detecting dead pixels

        Returns:
            result: Corrected bayer tensor of same shape as input
        """
        # Convert to int32 for calculations
        bayer = bayer.to(torch.int32)

        # Pad the input with reflection padding
        padded = F.pad(bayer, (2, 2, 2, 2), mode="reflect")

        # Get all 9 shifted versions for 3x3 window
        shifts = []
        for i in range(3):
            for j in range(3):
                shifts.append(
                    padded[:, :, i : i + bayer.shape[2], j : j + bayer.shape[3]]
                )

        # Center pixel is at index 4 (middle of 3x3 window)
        center = shifts[4]

        # Calculate mask for dead pixels
        mask = torch.ones_like(center, dtype=torch.bool)
        for i in [1, 7, 3, 5, 0, 2, 6, 8]:  # All neighbors
            mask &= torch.abs(center - shifts[i]) > diff_threshold

        # Calculate directional differences
        dv = torch.abs(2 * center - shifts[1] - shifts[7])  # Vertical
        dh = torch.abs(2 * center - shifts[3] - shifts[5])  # Horizontal
        ddl = torch.abs(2 * center - shifts[0] - shifts[8])  # Diagonal left
        ddr = torch.abs(2 * center - shifts[6] - shifts[2])  # Diagonal right

        # Stack differences and find minimum direction
        diffs = torch.stack([dv, dh, ddl, ddr], dim=-1)
        indices = torch.argmin(diffs, dim=-1).unsqueeze(-1)

        # Calculate neighbor averages
        neighbor_avgs = torch.stack(
            [
                (shifts[1] + shifts[7]) >> 1,  # Vertical average
                (shifts[3] + shifts[5]) >> 1,  # Horizontal average
                (shifts[0] + shifts[8]) >> 1,  # Diagonal left average
                (shifts[6] + shifts[2]) >> 1,  # Diagonal right average
            ],
            dim=-1,
        )

        # Select values based on minimum direction
        corrected = torch.gather(neighbor_avgs, -1, indices).squeeze(-1)

        # Combine original and corrected values using mask
        result = torch.where(mask, corrected, center)

        return result.to(torch.uint16)

    @staticmethod
    def demosaic(bayer, method="naive"):
        """Demosaic the image from RAW Bayer (H, W)/(B, 1, H, W) to RAW RGB (3, H, W)/(B, 3, H, W) using bilinear interpolation.

        Args:
            bayer (tensor): Bayer image, shape (H, W) or (B, 1, H, W), data range [0, 1].

        Returns:
            raw_rgb (tensor): RGB image, shape (3, H, W) or (B, 3, H, W), data range [0, 1].
        """
        if len(bayer.shape) == 2:
            bayer = bayer.unsqueeze(0).unsqueeze(0)
            single_image = True
        else:
            single_image = False

        B, _, H, W = bayer.shape
        raw_rgb = torch.zeros((B, 3, H, W), dtype=bayer.dtype, device=bayer.device)

        if method == "naive":
            # Pad bayer image to avoid out-of-boundary access in interpolation
            bayer_pad = F.pad(bayer, (1, 1, 1, 1), mode="replicate")

            # Green component
            raw_rgb[:, 1, 0:H:2, 0:W:2] = (
                bayer_pad[:, 0, 0 : H - 1 : 2, 1:W:2]
                + bayer_pad[:, 0, 1:H:2, 0 : W - 1 : 2]
                + bayer_pad[:, 0, 1:H:2, 2 : W + 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 1:W:2]
            ) / 4
            raw_rgb[:, 1, 0 : H - 1 : 2, 1:W:2] = bayer[:, 0, 0 : H - 1 : 2, 1:W:2]
            raw_rgb[:, 1, 1:H:2, 0 : W - 1 : 2] = bayer[:, 0, 1:H:2, 0 : W - 1 : 2]
            raw_rgb[:, 1, 1:H:2, 1:W:2] = (
                bayer_pad[:, 0, 1:H:2, 2 : W + 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 1:W:2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 3 : W + 2 : 2]
                + bayer_pad[:, 0, 3 : H + 2 : 2, 2 : W + 1 : 2]
            ) / 4

            # Red component
            raw_rgb[:, 0, 0 : H - 1 : 2, 0 : W - 1 : 2] = bayer[
                :, 0, 0 : H - 1 : 2, 0 : W - 1 : 2
            ]
            raw_rgb[:, 0, 0 : H - 1 : 2, 1:W:2] = (
                bayer_pad[:, 0, 1:H:2, 1:W:2] + bayer_pad[:, 0, 1:H:2, 3 : W + 2 : 2]
            ) / 2
            raw_rgb[:, 0, 1:H:2, 0 : W - 1 : 2] = (
                bayer_pad[:, 0, 1:H:2, 1:W:2] + bayer_pad[:, 0, 3 : H + 2 : 2, 1:W:2]
            ) / 2
            raw_rgb[:, 0, 1:H:2, 1:W:2] = (
                bayer_pad[:, 0, 1:H:2, 1:W:2]
                + bayer_pad[:, 0, 1:H:2, 3 : W + 2 : 2]
                + bayer_pad[:, 0, 3 : H + 2 : 2, 1:W:2]
                + bayer_pad[:, 0, 3 : H + 2 : 2, 3 : W + 2 : 2]
            ) / 4

            # Blue component
            raw_rgb[:, 2, 0 : H - 1 : 2, 0 : W - 1 : 2] = (
                bayer_pad[:, 0, 0 : H - 1 : 2, 0 : W - 1 : 2]
                + bayer_pad[:, 0, 0 : H - 1 : 2, 2 : W + 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 0 : W - 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 2 : W + 1 : 2]
            ) / 4
            raw_rgb[:, 2, 0 : H - 1 : 2, 1:W:2] = (
                bayer_pad[:, 0, 0 : H - 1 : 2, 2 : W + 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 2 : W + 1 : 2]
            ) / 2
            raw_rgb[:, 2, 1:H:2, 0 : W - 1 : 2] = (
                bayer_pad[:, 0, 2 : H + 1 : 2, 0 : W - 1 : 2]
                + bayer_pad[:, 0, 2 : H + 1 : 2, 2 : W + 1 : 2]
            ) / 2
            raw_rgb[:, 2, 1:H:2, 1:W:2] = bayer[:, 0, 1:H:2, 1:W:2]

        elif method == "3x3":
            # Create masks for red, green, and blue pixels according to RGGB pattern
            red_mask = torch.zeros_like(bayer)
            green_mask = torch.zeros_like(bayer)
            blue_mask = torch.zeros_like(bayer)

            # Red pixel mask (R) - top-left pixel of the 2x2 block
            red_mask[:, :, 0::2, 0::2] = 1

            # Green pixel masks (G) - top-right and bottom-left pixels of the 2x2 block
            green_mask[:, :, 0::2, 1::2] = 1  # Top-right green
            green_mask[:, :, 1::2, 0::2] = 1  # Bottom-left green

            # Blue pixel mask (B) - bottom-right pixel of the 2x2 block
            blue_mask[:, :, 1::2, 1::2] = 1

            # Extract known color values
            raw_rgb[:, 0, :, :] = (bayer * red_mask).squeeze(1)  # Red channel
            raw_rgb[:, 1, :, :] = (bayer * green_mask).squeeze(1)  # Green channel
            raw_rgb[:, 2, :, :] = (bayer * blue_mask).squeeze(1)  # Blue channel

            # Define interpolation kernels
            kernel_G = (
                torch.tensor(
                    [[0, 1 / 4, 0], [1 / 4, 1, 1 / 4], [0, 1 / 4, 0]],
                    dtype=bayer.dtype,
                    device=bayer.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            kernel_RB = (
                torch.tensor(
                    [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]],
                    dtype=bayer.dtype,
                    device=bayer.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            # Interpolate green channel
            mask_G = (green_mask > 0).float()
            raw_rgb_G = raw_rgb[:, 1, :, :].unsqueeze(1)
            raw_rgb[:, 1, :, :] = (
                F.conv2d(raw_rgb_G * mask_G, kernel_G, padding=1)
                / F.conv2d(mask_G, kernel_G, padding=1)
            ).squeeze()

            # Interpolate red channel
            mask_R = (red_mask > 0).float()
            raw_rgb_R = raw_rgb[:, 0, :, :].unsqueeze(1)
            raw_rgb[:, 0, :, :] = (
                F.conv2d(raw_rgb_R * mask_R, kernel_RB, padding=1)
                / F.conv2d(mask_R, kernel_RB, padding=1)
            ).squeeze()

            # Interpolate blue channel
            mask_B = (blue_mask > 0).float()
            raw_rgb_B = raw_rgb[:, 2, :, :].unsqueeze(1)
            raw_rgb[:, 2, :, :] = (
                F.conv2d(raw_rgb_B * mask_B, kernel_RB, padding=1)
                / F.conv2d(mask_B, kernel_RB, padding=1)
            ).squeeze()
        else:
            raise ValueError("Invalid demosaic method")

        if single_image:
            raw_rgb = raw_rgb.squeeze(0)

        return raw_rgb

    @staticmethod
    def gamma_correction(img, gamma_param=2.2, quantize=False):
        """Gamma correction.

        Reference:
            [1] Architectural Analysis of a Baseline ISP Pipeline. (page 34 and 51)

        Args:
            img (tensor): Input image. Shape of [B, C, H, W].
            gamma_param (float): Gamma parameter.
            quantize (bool): Whether to quantize the image to 8-bit.

        Returns:
            img_gamma (tensor): Gamma corrected image. Shape of [B, C, H, W].
        """
        img_gamma = torch.pow(torch.clamp(img, min=1e-8), 1 / gamma_param)
        if quantize:
            img_gamma = torch.round(img_gamma * 255) / 255
        return img_gamma

    @staticmethod
    def rgb_to_ycrcb(rgb_image):
        """Color Space Conversion (CSC) from RGB to YCrCb."""
        return self.csc(rgb_image)

    @staticmethod
    def csc(rgb_image):
        """Color Space Conversion (CSC) from RGB to YCrCb. Converting to YCrCb makes the downstream processing more efficient.

        Reference:
            [1] Architectural Analysis of a Baseline ISP Pipeline. (page 38)

        Args:
            rgb_image (tensor): Input tensor of shape [B, 3, H, W] in RGB format. Data range [0, 255].

        Returns:
            y_image (tensor): Luminance component [B, 1, H, W]. Data range [0, 255].
            cbcr_image (tensor): Chrominance components [B, 2, H, W]. Data range [0, 255].
        """
        # Conversion matrix (x256)
        matrix = torch.tensor(
            [[66, -38, 112], [129, -74, -94], [25, 112, -18]],
            dtype=torch.int32,
            device=rgb_image.device,
        ).T  # Transpose to match input format

        # Bias terms
        bias = torch.tensor(
            [16, 128, 128], dtype=torch.int32, device=rgb_image.device
        ).view(1, 3, 1, 1)

        # Convert input to int32 for calculations
        rgb_int = rgb_image.to(torch.int32)

        # Reshape and perform matrix multiplication
        B, C, H, W = rgb_int.shape
        rgb_reshaped = rgb_int.view(B, C, -1)  # [B, 3, H*W]

        # Matrix multiplication and right shift by 8 (divide by 256)
        ycrcb = (matrix @ rgb_reshaped) >> 8  # [B, 3, H*W]

        # Reshape back and add bias
        ycrcb = ycrcb.view(B, 3, H, W)
        ycrcb = ycrcb + bias

        # Clip values to uint8 range and convert
        ycrcb = torch.clamp(ycrcb, 0, 255).to(torch.uint8)

        # Split into Y and CbCr components
        y_image = ycrcb[:, 0:1]  # Keep dimension for channel
        cbcr_image = ycrcb[:, 1:]

        return y_image, cbcr_image

    @staticmethod
    def ycbcr_to_rgb(ycbcr_tensor):
        """Convert YCbCr 3-channel tensor into sRGB tensor

        Args:
            ycbcr_tensor (torch.Tensor): Input YCbCr tensor with values in [0, 255]
                Shape can be either (H, W, 3) or (B, 3, H, W)

        Returns:
            torch.Tensor: RGB tensor with values in [0, 255]
        """
        # Check if input needs channel dimension permutation
        need_permute = ycbcr_tensor.shape[-3] == 3
        if need_permute:
            ycbcr_tensor = (
                ycbcr_tensor.permute(0, 2, 3, 1)
                if ycbcr_tensor.dim() == 4
                else ycbcr_tensor.permute(1, 2, 0)
            )

        # Convert to int32 equivalent
        ycbcr_tensor = ycbcr_tensor.to(torch.float32)

        # Define conversion matrix and bias (x256)
        matrix = torch.tensor(
            [[298, 0, 409], [298, -100, -208], [298, 516, 0]],
            dtype=torch.float32,
            device=ycbcr_tensor.device,
        ).T

        bias = torch.tensor(
            [-56992, 34784, -70688], dtype=torch.float32, device=ycbcr_tensor.device
        ).reshape(1, 1, 3)

        # Perform conversion
        rgb_tensor = (ycbcr_tensor @ matrix + bias) / 256.0
        rgb_tensor = torch.clamp(rgb_tensor, 0, 255)

        # Restore original dimension order if needed
        if need_permute:
            rgb_tensor = (
                rgb_tensor.permute(0, 3, 1, 2)
                if rgb_tensor.dim() == 4
                else rgb_tensor.permute(2, 0, 1)
            )

        return rgb_tensor

"""Demosaic, or Color Filter Array (CFA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Demosaic(nn.Module):
    """Demosaic, or Color Filter Array (CFA).
    
    Converts a Bayer pattern image to a full RGB image by interpolating
    missing color values at each pixel location.
    
    Supported methods:
        - "bilinear": Simple bilinear interpolation (fast, lower quality)
        - "malvar": Malvar-He-Cutler high-quality gradient-corrected interpolation
    
    Reference:
        [1] Malvar, He, Cutler. "High-Quality Linear Interpolation for Demosaicing
            of Bayer-Patterned Color Images", ICASSP 2004.
    """

    def __init__(self, bayer_pattern="rggb", method="malvar"):
        """Initialize demosaic.

        Args:
            bayer_pattern: Bayer pattern, "rggb" or "bggr".
            method: Demosaic method, "bilinear" or "malvar".
        """
        super().__init__()
        self.bayer_pattern = bayer_pattern
        self.method = method
        
        # Pre-compute Malvar kernels if using that method
        if method == "malvar":
            self._init_malvar_kernels()

    def _init_malvar_kernels(self):
        """Initialize Malvar-He-Cutler demosaic kernels.
        
        These 5x5 kernels perform gradient-corrected bilinear interpolation
        to reduce color artifacts at edges.
        
        Reference:
            Malvar, He, Cutler. "High-Quality Linear Interpolation for Demosaicing
            of Bayer-Patterned Color Images", ICASSP 2004.
            https://www.ipol.im/pub/art/2011/g_mhcd/
        """
        # Kernel for G at R locations and G at B locations (same kernel)
        # This interpolates Green at Red or Blue pixel positions
        kernel_g_at_rb = torch.tensor([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2,  4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=torch.float32) / 8.0
        
        # Kernel for R at G in R row, B column (Gr positions)
        # and B at G in B row, R column (Gb positions)
        kernel_rb_at_g_rbcol = torch.tensor([
            [ 0,  0,  0.5,  0,  0],
            [ 0, -1,  0,   -1,  0],
            [-1,  4,  5,    4, -1],
            [ 0, -1,  0,   -1,  0],
            [ 0,  0,  0.5,  0,  0],
        ], dtype=torch.float32) / 8.0
        
        # Kernel for R at G in B row, R column (Gb positions)  
        # and B at G in R row, B column (Gr positions)
        kernel_rb_at_g_rbrow = torch.tensor([
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  4, -1,  0],
            [0.5, 0,  5,  0, 0.5],
            [ 0, -1,  4, -1,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=torch.float32) / 8.0
        
        # Kernel for R at B locations and B at R locations
        kernel_rb_at_br = torch.tensor([
            [ 0,  0, -1.5,  0,   0],
            [ 0,  2,  0,    2,   0],
            [-1.5, 0, 6,    0, -1.5],
            [ 0,  2,  0,    2,   0],
            [ 0,  0, -1.5,  0,   0],
        ], dtype=torch.float32) / 8.0
        
        # Register as buffers (moved to device with model)
        self.register_buffer("malvar_g_at_rb", kernel_g_at_rb.view(1, 1, 5, 5))
        self.register_buffer("malvar_rb_at_g_rbcol", kernel_rb_at_g_rbcol.view(1, 1, 5, 5))
        self.register_buffer("malvar_rb_at_g_rbrow", kernel_rb_at_g_rbrow.view(1, 1, 5, 5))
        self.register_buffer("malvar_rb_at_br", kernel_rb_at_br.view(1, 1, 5, 5))

    def _malvar_demosaic(self, bayer):
        """Malvar-He-Cutler high-quality demosaic method (differentiable).
        
        Uses gradient-corrected 5x5 kernels to interpolate missing colors
        while preserving edges better than simple bilinear interpolation.
        
        RGGB Bayer pattern layout:
            (0,0) R   (0,1) Gr  (0,2) R   (0,3) Gr
            (1,0) Gb  (1,1) B   (1,2) Gb  (1,3) B
            (2,0) R   (2,1) Gr  (2,2) R   (2,3) Gr
            (3,0) Gb  (3,1) B   (3,2) Gb  (3,3) B

        Args:
            bayer (torch.Tensor): Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            raw_rgb (torch.Tensor): Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        B, C, H, W = bayer.shape
        
        # Pad the bayer image for 5x5 kernel (2 pixels on each side)
        bayer_pad = F.pad(bayer, (2, 2, 2, 2), mode="reflect")
        
        # Create masks for each pixel type in RGGB pattern
        # These masks indicate where each color is sampled
        r_mask = torch.zeros((1, 1, H, W), device=bayer.device, dtype=bayer.dtype)
        gr_mask = torch.zeros((1, 1, H, W), device=bayer.device, dtype=bayer.dtype)
        gb_mask = torch.zeros((1, 1, H, W), device=bayer.device, dtype=bayer.dtype)
        b_mask = torch.zeros((1, 1, H, W), device=bayer.device, dtype=bayer.dtype)
        
        r_mask[:, :, 0::2, 0::2] = 1    # Red at (even, even)
        gr_mask[:, :, 0::2, 1::2] = 1   # Green-Red row at (even, odd)
        gb_mask[:, :, 1::2, 0::2] = 1   # Green-Blue row at (odd, even)
        b_mask[:, :, 1::2, 1::2] = 1    # Blue at (odd, odd)
        
        g_mask = gr_mask + gb_mask  # All green positions
        
        # Apply Malvar kernels via convolution
        # G at R and B locations
        g_at_rb = F.conv2d(bayer_pad, self.malvar_g_at_rb.to(bayer.dtype), padding=0)
        
        # R at Gr locations (R row, B col) - use horizontal kernel
        r_at_gr = F.conv2d(bayer_pad, self.malvar_rb_at_g_rbcol.to(bayer.dtype), padding=0)
        
        # R at Gb locations (B row, R col) - use vertical kernel
        r_at_gb = F.conv2d(bayer_pad, self.malvar_rb_at_g_rbrow.to(bayer.dtype), padding=0)
        
        # R at B locations
        r_at_b = F.conv2d(bayer_pad, self.malvar_rb_at_br.to(bayer.dtype), padding=0)
        
        # B at Gr locations (R row, B col) - use vertical kernel
        b_at_gr = F.conv2d(bayer_pad, self.malvar_rb_at_g_rbrow.to(bayer.dtype), padding=0)
        
        # B at Gb locations (B row, R col) - use horizontal kernel
        b_at_gb = F.conv2d(bayer_pad, self.malvar_rb_at_g_rbcol.to(bayer.dtype), padding=0)
        
        # B at R locations
        b_at_r = F.conv2d(bayer_pad, self.malvar_rb_at_br.to(bayer.dtype), padding=0)
        
        # Assemble the RGB channels
        # Red channel: R at R (original) + R at Gr + R at Gb + R at B
        red = (bayer * r_mask + 
               r_at_gr * gr_mask + 
               r_at_gb * gb_mask + 
               r_at_b * b_mask)
        
        # Green channel: G at Gr + G at Gb (original) + G at R + G at B
        green = (bayer * g_mask + 
                 g_at_rb * r_mask + 
                 g_at_rb * b_mask)
        
        # Blue channel: B at B (original) + B at Gr + B at Gb + B at R
        blue = (bayer * b_mask + 
                b_at_gr * gr_mask + 
                b_at_gb * gb_mask + 
                b_at_r * r_mask)
        
        # Stack channels
        raw_rgb = torch.cat([red, green, blue], dim=1)
        
        # Clamp to valid range (kernel interpolation can exceed [0, 1])
        raw_rgb = torch.clamp(raw_rgb, 0.0, 1.0)
        
        return raw_rgb

    def _bilinear_demosaic(self, bayer):
        """Bilinear interpolation demosaic method.

        RGGB Bayer pattern layout:
            (0,0) R  (0,1) G  (0,2) R  (0,3) G
            (1,0) G  (1,1) B  (1,2) G  (1,3) B
            (2,0) R  (2,1) G  (2,2) R  (2,3) G
            (3,0) G  (3,1) B  (3,2) G  (3,3) B

        Args:
            bayer (torch.Tensor): Input tensor of shape [B, 1, H, W], data range [0, 1].

        Returns:
            raw_rgb (torch.Tensor): Output tensor of shape [B, 3, H, W], data range [0, 1].
        """
        B, C, H, W = bayer.shape
        raw_rgb = torch.zeros((B, 3, H, W), device=bayer.device, dtype=bayer.dtype)

        # Pad the bayer image for boundary handling (1 pixel on each side)
        bayer_pad = F.pad(bayer, (1, 1, 1, 1), mode="reflect")

        # --- Red channel ---
        # R at R positions (0,0): direct copy
        raw_rgb[:, 0, 0::2, 0::2] = bayer[:, 0, 0::2, 0::2]

        # R at Gr positions (0,1): average of left and right R neighbors
        # In padded coords, original (0,1) is at (1,2)
        # Left R is at (1,1), Right R is at (1,3) -> average R from col 0 and col 2
        raw_rgb[:, 0, 0::2, 1::2] = (
            bayer_pad[:, 0, 1 : H + 1 : 2, 0:W:2]
            + bayer_pad[:, 0, 1 : H + 1 : 2, 2 : W + 2 : 2]
        ) / 2

        # R at Gb positions (1,0): average of top and bottom R neighbors
        raw_rgb[:, 0, 1::2, 0::2] = (
            bayer_pad[:, 0, 0:H:2, 1 : W + 1 : 2]
            + bayer_pad[:, 0, 2 : H + 2 : 2, 1 : W + 1 : 2]
        ) / 2

        # R at B positions (1,1): average of four diagonal R neighbors
        raw_rgb[:, 0, 1::2, 1::2] = (
            (
                bayer_pad[:, 0, 0:H:2, 0:W:2]  # top-left
                + bayer_pad[:, 0, 0:H:2, 2 : W + 2 : 2]  # top-right
                + bayer_pad[:, 0, 2 : H + 2 : 2, 0:W:2]  # bottom-left
                + bayer_pad[:, 0, 2 : H + 2 : 2, 2 : W + 2 : 2]  # bottom-right
            )
            / 4
        )

        # --- Green channel ---
        # G at Gr positions (0,1): direct copy
        raw_rgb[:, 1, 0::2, 1::2] = bayer[:, 0, 0::2, 1::2]

        # G at Gb positions (1,0): direct copy
        raw_rgb[:, 1, 1::2, 0::2] = bayer[:, 0, 1::2, 0::2]

        # G at R positions (0,0): average of four orthogonal G neighbors
        raw_rgb[:, 1, 0::2, 0::2] = (
            (
                bayer_pad[
                    :, 0, 0:H:2, 1 : W + 1 : 2
                ]  # top (Gr from previous row or reflected)
                + bayer_pad[:, 0, 2 : H + 2 : 2, 1 : W + 1 : 2]  # bottom (Gb)
                + bayer_pad[
                    :, 0, 1 : H + 1 : 2, 0:W:2
                ]  # left (Gb from previous col or reflected)
                + bayer_pad[:, 0, 1 : H + 1 : 2, 2 : W + 2 : 2]  # right (Gr)
            )
            / 4
        )

        # G at B positions (1,1): average of four orthogonal G neighbors
        raw_rgb[:, 1, 1::2, 1::2] = (
            (
                bayer_pad[:, 0, 1 : H + 1 : 2, 2 : W + 2 : 2]  # top (Gr)
                + bayer_pad[
                    :, 0, 3 : H + 3 : 2, 2 : W + 2 : 2
                ]  # bottom (Gr from next row)
                + bayer_pad[:, 0, 2 : H + 2 : 2, 1 : W + 1 : 2]  # left (Gb)
                + bayer_pad[
                    :, 0, 2 : H + 2 : 2, 3 : W + 3 : 2
                ]  # right (Gb from next col)
            )
            / 4
        )

        # --- Blue channel ---
        # B at B positions (1,1): direct copy
        raw_rgb[:, 2, 1::2, 1::2] = bayer[:, 0, 1::2, 1::2]

        # B at Gr positions (0,1): average of top and bottom B neighbors
        raw_rgb[:, 2, 0::2, 1::2] = (
            bayer_pad[:, 0, 0:H:2, 2 : W + 2 : 2]
            + bayer_pad[:, 0, 2 : H + 2 : 2, 2 : W + 2 : 2]
        ) / 2

        # B at Gb positions (1,0): average of left and right B neighbors
        raw_rgb[:, 2, 1::2, 0::2] = (
            bayer_pad[:, 0, 2 : H + 2 : 2, 0:W:2]
            + bayer_pad[:, 0, 2 : H + 2 : 2, 2 : W + 2 : 2]
        ) / 2

        # B at R positions (0,0): average of four diagonal B neighbors
        raw_rgb[:, 2, 0::2, 0::2] = (
            (
                bayer_pad[:, 0, 0:H:2, 0:W:2]  # top-left
                + bayer_pad[:, 0, 0:H:2, 2 : W + 2 : 2]  # top-right
                + bayer_pad[:, 0, 2 : H + 2 : 2, 0:W:2]  # bottom-left
                + bayer_pad[:, 0, 2 : H + 2 : 2, 2 : W + 2 : 2]  # bottom-right
            )
            / 4
        )

        return raw_rgb

    def forward(self, bayer):
        """Demosaic a Bayer pattern image to RGB.

        Args:
            bayer: Input tensor of shape [B, 1, H, W].

        Returns:
            rgb: Output tensor of shape [B, 3, H, W].
        """
        if bayer.dim() == 3:
            bayer = bayer.unsqueeze(0)
            batch_dim = False
        else:
            batch_dim = True

        if self.method == "bilinear":
            raw_rgb = self._bilinear_demosaic(bayer)
        elif self.method == "malvar":
            raw_rgb = self._malvar_demosaic(bayer)
        else:
            raise ValueError(f"Invalid demosaic method: {self.method}. Use 'bilinear' or 'malvar'.")

        if not batch_dim:
            raw_rgb = raw_rgb.squeeze(0)

        return raw_rgb

    def reverse(self, img):
        """Inverse demosaic from RAW RGB to RAW Bayer.

        Args:
            img (torch.Tensor): RAW RGB image, shape [3, H, W] or [B, 3, H, W], data range [0, 1].

        Returns:
            torch.Tensor: Bayer image, shape [1, H, W] or [B, 1, H, W], data range [0, 1].
        """
        if img.ndim == 3:
            # Input shape: [3, H, W]
            batch_dim = False
            C, H, W = img.shape
        elif img.ndim == 4:
            # Input shape: [B, 3, H, W]
            batch_dim = True
            B, C, H, W = img.shape
        else:
            raise ValueError(
                "Input image must have 3 or 4 dimensions corresponding to [3, H, W] or [B, 3, H, W]."
            )

        if C != 3:
            raise ValueError("Input image must have 3 channels corresponding to RGB.")

        if batch_dim:
            bayer = torch.zeros((B, 1, H, W), dtype=img.dtype, device=img.device)
            bayer[:, 0, 0::2, 0::2] = img[:, 0, 0::2, 0::2]
            bayer[:, 0, 0::2, 1::2] = img[:, 1, 0::2, 1::2]
            bayer[:, 0, 1::2, 0::2] = img[:, 1, 1::2, 0::2]
            bayer[:, 0, 1::2, 1::2] = img[:, 2, 1::2, 1::2]
        else:
            bayer = torch.zeros((1, H, W), dtype=img.dtype, device=img.device)
            bayer[0, 0::2, 0::2] = img[0, 0::2, 0::2]
            bayer[0, 0::2, 1::2] = img[1, 0::2, 1::2]
            bayer[0, 1::2, 0::2] = img[1, 1::2, 0::2]
            bayer[0, 1::2, 1::2] = img[2, 1::2, 1::2]

        return bayer

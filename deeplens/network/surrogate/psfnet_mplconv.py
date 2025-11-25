# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""A MLP-Conv network architecture to represent the spatiallly varying PSF of a lens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseNormalization(nn.Module):
    def __init__(self):
        super(ChannelwiseNormalization, self).__init__()

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        # Reshape to [batch, channels, -1] to apply softmax over spatial dimensions
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        # Apply softmax along the last dimension (spatial locations)
        x_softmax = F.softmax(x_flat, dim=2)
        # Reshape back to original [batch, channels, height, width]
        return x_softmax.view(b, c, h, w)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv should have stride=1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut unchanged
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = ResidualBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.residual(x)  # Refine first
        x = self.upsample(x)  # Then upsample
        x = self.norm(x)
        return self.activation(x)


class MLPConditioner(nn.Module):
    """
    MLP to process input (r, z) into a latent vector.
    Input: [batch_size, in_chan]
    Output: [batch_size, latent_dim] (flat vector)
    """

    def __init__(self, in_chan=2, latent_dim=4096):
        super(MLPConditioner, self).__init__()
        # Learnable scaling and shifting parameters to handle different input ranges
        self.scale = nn.Parameter(torch.ones(in_chan))
        self.shift = nn.Parameter(torch.zeros(in_chan))
        self.fc = nn.Sequential(
            nn.Linear(in_chan, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        x = x * self.scale + self.shift
        return self.fc(x)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder to generate PSF from latent vector with multi-scale features.
    Input: [batch_size, latent_dim] (flat)
    Output: [batch_size, out_chan, kernel_size, kernel_size]
    Assumes kernel_size=128 and latent reshapes to [64, 16, 16].
    """

    def __init__(
        self, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    ):
        super(ConvDecoder, self).__init__()
        # Validate latent dim matches reshape
        self.initial_height = (
            16  # Starting height/width for upsampling (16 -> 32 -> 64 -> 128)
        )
        self.initial_shape = (latent_channels, self.initial_height, self.initial_height)
        expected_dim = latent_channels * self.initial_height * self.initial_height
        assert latent_dim == expected_dim, (
            f"Latent dim must be {expected_dim} for reshape, got {latent_dim}"
        )

        # If kernel_size changes, adjust the number of upsample layers
        assert kernel_size == self.initial_height * (2**3), (
            f"Adjust upsample layers for kernel_size={kernel_size}"
        )

        # Decoder blocks as individual modules for multi-scale access
        self.decoder_block1 = DecoderBlock(latent_channels, 32)  # 16x16 -> 32x32
        self.decoder_block2 = DecoderBlock(32, 16)  # 32x32 -> 64x64
        self.decoder_block3 = DecoderBlock(16, 8)  # 64x64 -> 128x128

        # Skip connections for multi-scale features
        self.skip_conv1 = nn.Conv2d(32, 8, 1)  # From 32x32 level
        self.skip_conv2 = nn.Conv2d(16, 8, 1)  # From 64x64 level

        # Final layers
        self.final_conv = nn.Conv2d(8, out_chan, kernel_size=3, padding=1)
        self.normalization = ChannelwiseNormalization()

    def forward(self, latent):
        batch_size = latent.size(0)
        # Reshape flat latent to initial feature map
        x = latent.view(batch_size, *self.initial_shape)

        # Store intermediate features for multi-scale processing
        x = self.decoder_block1(x)  # 32x32, 32 channels
        skip1 = F.interpolate(
            self.skip_conv1(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block2(x)  # 64x64, 16 channels
        skip2 = F.interpolate(
            self.skip_conv2(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block3(x)  # 128x128, 8 channels

        # Combine multi-scale features
        x = x + skip1 + skip2

        # Final processing
        x = self.final_conv(x)
        return self.normalization(x)


class PSFNet_MLPConv(nn.Module):
    """
    Combined model: MLPConditioner + ConvDecoder.
    Input: [batch_size, 2] (r, z)
    Output: [batch_size, out_chan, kernel_size, kernel_size]
    """

    def __init__(
        self,
        in_chan=2,
        kernel_size=128,
        out_chan=3,
        latent_dim=4096,
        latent_channels=16,
    ):
        super(PSFNet_MLPConv, self).__init__()
        self.mlp = MLPConditioner(in_chan=in_chan, latent_dim=latent_dim)
        self.decoder = ConvDecoder(
            kernel_size=kernel_size,
            out_chan=out_chan,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
        )

    def forward(self, x):
        psf = self.decoder(self.mlp(x))
        return psf


# Test code
if __name__ == "__main__":
    # Instantiate the model
    model = PSFNet_MLPConv(
        in_chan=2, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    )

    # Dummy input: batch_size=2, with example (r, z) values
    # r in [-1,1], z in [-10000,0]
    rz = torch.tensor(
        [
            [0.5, -5000.0],  # Example 1
            [-0.3, -2000.0],  # Example 2
        ]
    )  # Shape: [2, 2]

    # Forward pass
    with torch.no_grad():  # No gradients for testing
        psf_output = model(rz)

    # Print shapes and a sample value
    print(f"Input shape: {rz.shape}")
    print(f"Output shape: {psf_output.shape}")  # Should be [2, 3, 128, 128]

    # Check if output sums to ~1 per channel (if using Softmax instead)
    print(f"Sum per channel (first batch): {psf_output[0].sum(dim=(1, 2))}")

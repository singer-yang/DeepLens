import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        x = x * self.scale + self.shift
        return self.fc(x)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder to generate PSF from latent vector.
    Input: [batch_size, latent_dim] (flat)
    Output: [batch_size, out_chan, kernel_size, kernel_size]
    Assumes kernel_size=64 and latent reshapes to [64, 8, 8].
    """

    def __init__(self, kernel_size=64, out_chan=3, latent_dim=4096, latent_channels=64):
        super(ConvDecoder, self).__init__()
        # Validate latent dim matches reshape
        self.initial_height = (
            8  # Starting height/width for upsampling (8 -> 16 -> 32 -> 64)
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

        self.decoder = nn.Sequential(
            # Upsample 8x8 -> 16x16
            DecoderBlock(latent_channels, 32),
            # Upsample 16x16 -> 32x32
            DecoderBlock(32, 16),
            # Upsample 32x32 -> 64x64
            DecoderBlock(16, 8),
            # Final conv to 3 channels (no upsample)
            nn.Conv2d(8, out_chan, kernel_size=3, padding=1),
            ChannelwiseNormalization(),
        )

    def forward(self, latent):
        batch_size = latent.size(0)
        # Reshape flat latent to initial feature map
        x = latent.view(batch_size, *self.initial_shape)
        return self.decoder(x)


class PSFNet_MLPConv(nn.Module):
    """
    Combined model: MLPConditioner + ConvDecoder.
    Input: [batch_size, 2] (r, z)
    Output: [batch_size, out_chan, kernel_size, kernel_size]
    """

    def __init__(
        self, in_chan=2, kernel_size=64, out_chan=3, latent_dim=4096, latent_channels=64
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
        assert x[:, 0].abs().max() <= 1.1, "r should be in [-1, 1]"
        assert x[:, 1].max() <= 0, "z should be <= 0"

        latent = self.mlp(x)
        psf = self.decoder(latent)
        return psf


def test_inference_time():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PSFNet_MLPConv(
        in_chan=2, kernel_size=64, out_chan=3, latent_dim=4096, latent_channels=64
    )
    model.to(device)
    model.half()  # Convert model to float16
    model.eval()

    # Input tensor of shape [N, 2]
    # r in [-1, 1], z <= 0
    N = 200000
    r = 2 * torch.rand(N, 1) - 1  # Random values between -1 and 1
    z = -10000 * torch.rand(N, 1)  # Random values between -10000 and 0
    input_tensor = (
        torch.cat((r, z), dim=1).to(device).half()
    )  # Convert input to float16

    with torch.no_grad():
        # Warm-up run
        _ = model(input_tensor[:10])

        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        output = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time for input shape [{N}, 2]: {inference_time:.4f} seconds")
    print(f"Output shape: {output.shape}")


# Test code
if __name__ == "__main__":
    # Test inference time
    test_inference_time()
    breakpoint()

    # Instantiate the model
    model = PSFNet_MLPConv(
        in_chan=2, kernel_size=64, out_chan=3, latent_dim=4096, latent_channels=64
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
    print(f"Output shape: {psf_output.shape}")  # Should be [2, 3, 64, 64]

    # Check if output sums to ~1 per channel (if using Softmax instead)
    print(f"Sum per channel (first batch): {psf_output[0].sum(dim=(1, 2))}")

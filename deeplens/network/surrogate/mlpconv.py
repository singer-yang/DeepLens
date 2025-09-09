import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConv(nn.Module):
    """MLP encoder + convolutional decoder proposed in "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design". This network suits for high-k intensity/amplitude PSF function prediction.

    Input:
        in_features (int): Input features, shape of [batch_size, in_features].
        ks (int): The size of the output image.
        channels (int): The number of output channels. Defaults to 3.
        activation (str): The activation function. Defaults to 'relu'.

    Output:
        x (Tensor): The output image. Shape of [batch_size, channels, ks, ks].
    """

    def __init__(self, in_features, ks, channels=3, activation="relu"):
        super(MLPConv, self).__init__()

        self.ks_mlp = min(ks, 32)
        if ks > 32:
            assert ks % 32 == 0, "ks must be 32n"
            upsample_times = int(math.log(ks / 32, 2))

        linear_output = channels * self.ks_mlp**2
        self.ks = ks
        self.channels = channels

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, linear_output),
        )

        # Conv decoder
        conv_layers = []
        conv_layers.append(
            nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        for _ in range(upsample_times):
            conv_layers.append(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Upsample(scale_factor=2))

        conv_layers.append(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*conv_layers)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encode the input using the MLP
        encoded = self.encoder(x)

        # Reshape the output from the MLP to feed to the CNN
        decoded_input = encoded.view(
            -1, self.channels, self.ks_mlp, self.ks_mlp
        )  # reshape to (batch_size, channels, height, width)

        # Decode the output using the CNN
        decoded = self.decoder(decoded_input)

        # This normalization only works for PSF network
        decoded = nn.Sigmoid()(decoded)
        decoded = F.normalize(decoded, p=1, dim=[-1, -2])

        return decoded


if __name__ == "__main__":
    # Test case
    # Create a model with 4 input features and a 64x64 output
    model = MLPConv(in_features=4, ks=64, channels=3)

    # Create a dummy input tensor with batch size 1 and 4 features
    # Shape: [batch_size, in_features]
    input_tensor = torch.randn(1, 4)

    # Get the model output
    output_tensor = model(input_tensor)

    # Print the shapes
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # Verify the output shape
    # Expected shape: [batch_size, channels, ks, ks]
    assert output_tensor.shape == (1, 3, 64, 64)
    print("Test passed!")



import math
import torch.nn as nn
import torch.nn.functional as nnF

class MLPConv(nn.Module):
    """ MLP encoder + convolutional decoder proposed in "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design". This network suits for high-k intensity/amplitude PSF function prediction.

    Input:
        in_features (int): Input features, shape of [batch_size, in_features].
        ks (int): The size of the output image.
        channels (int): The number of output channels. Defaults to 3.
        activation (str): The activation function. Defaults to 'relu'.

    Output:
        x (Tensor): The output image. Shape of [batch_size, channels, ks, ks].
    """
    def __init__(self, in_features, ks, channels=3, activation='relu'):
        super(MLPConv, self).__init__()

        self.ks_mlp = min(ks, 32)
        if ks > 32:
            assert ks % 32 == 0, 'ks must be 32n'
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
            nn.Linear(512, linear_output)
        )

        # Conv decoder
        conv_layers = []
        conv_layers.append(nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.ReLU())
        for _ in range(upsample_times):
            conv_layers.append(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Upsample(scale_factor=2))
        
        conv_layers.append(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*conv_layers)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encode the input using the MLP
        encoded = self.encoder(x)
        
        # Reshape the output from the MLP to feed to the CNN
        decoded_input = encoded.view(-1, self.channels, self.ks_mlp, self.ks_mlp)  # reshape to (batch_size, channels, height, width)
        
        # Decode the output using the CNN
        decoded = self.decoder(decoded_input)
        # decoded = self.activation(decoded)
        
        # This normalization only works for PSF network
        decoded = nnF.normalize(decoded, p=1, dim=[-1,-2])
        
        return decoded
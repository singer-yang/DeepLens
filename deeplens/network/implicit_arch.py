""" 
Commonly used network architectures for PSF estimation network. 

TODO: merge this file with network_surrogate.py in the future.

Challenges: 
(1), differen PSF kernel size, but the same physical size. Maybe we need to use conditional Siren, with unit physical size.
(2), how to accurately represent PSF kernel? and also smaller model size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as F

import math
import numpy as np
from einops import rearrange


# ==================================================================================================
# MLP networks
#   (1): MLP-only architecture
#   (2): MLP + Conv architecture
# ==================================================================================================


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class MLP(nn.Module):
    """ All-linear layer. This network suits for low-k intensity/amplitude PSF function prediction.
    """
    def __init__(self, in_features, out_features, hidden_features=64, hidden_layers=3):
        super(MLP, self).__init__()

        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features//4, bias=True))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features//4, hidden_features, bias=True))
        self.net.append(nn.ReLU(inplace=True))
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features, bias=True))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features, bias=True))
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

        # TODO: initialize weights

    def forward(self, x):
        x = self.net(x)
        x = nnF.normalize(x, p=1, dim=-1)
        return x
    

class MLPConv(nn.Module):
    """ MLP encoder + convolutional decoder proposed in "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design".

    This network suits for high-k intensity/amplitude PSF function prediction.
    """
    def __init__(self, in_features, ks, activation='relu', channels=1):
        super(MLPConv, self).__init__()

        assert ks % 4 == 0, 'ks must be 4n'
        self.ks_mlp = ks // 4
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
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1),
        )

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
        decoded = self.activation(decoded)
        
        return decoded
    





# ==================================================================================================
# Siren networks
#   (1) Siren
#   (2) Conditional Siren, 
#       (2.1) Concatenation
#       (2.2) Skip connection, copy from https://github.com/lucidrains/siren-pytorch/tree/master
# ==================================================================================================

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        # if exists(bias):
        #     bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = nnF.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out
    

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, outermost_linear=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            # self.layers.append(Siren(
            #     dim_in = layer_dim_in,
            #     dim_out = dim_hidden,
            #     w0 = layer_w0,
            #     use_bias = use_bias,
            #     is_first = is_first
            # ))

            self.layers.append(SineLayer(
                in_features = layer_dim_in,
                out_features = dim_hidden,
                omega_0 = layer_w0,
                bias = use_bias,
                is_first = is_first
            ))

        if outermost_linear:
            self.last_layer = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                # w_std = math.sqrt(6 / dim_hidden) / w0
                # self.last_layer.weight.uniform_(- w_std, w_std)
                nn.init.kaiming_normal_(self.last_layer.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        else:
            final_activation = nn.Identity() if not exists(final_activation) else final_activation
            self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)
        

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

            with torch.no_grad():
                # self.layers[-1][0].weight.uniform_(-1 / dim_hidden, 1 / dim_hidden)
                nn.init.kaiming_normal_(self.layers[-1][0].weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim = -1) # FIXME: I add dim = -1

        return tuple(hiddens)
    

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return nnF.mse_loss(img, out)

        return out
    

class ModulateSiren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dim_latent, num_layers, image_width, image_height, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, outermost_linear=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.img_width = image_width
        self.img_height = image_height
        
        # ==> Synthesizer
        synthesizer_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            synthesizer_layers.append(SineLayer(
                in_features = layer_dim_in,
                out_features = dim_hidden,
                omega_0 = layer_w0,
                bias = use_bias,
                is_first = is_first
            ))

        if outermost_linear:
            last_layer = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                # w_std = math.sqrt(6 / dim_hidden) / w0
                # self.last_layer.weight.uniform_(- w_std, w_std)
                nn.init.kaiming_normal_(last_layer.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        else:
            final_activation = nn.Identity() if not exists(final_activation) else final_activation
            last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)
        synthesizer_layers.append(last_layer)
        
        self.synthesizer = synthesizer_layers
        # self.synthesizer = nn.Sequential(*synthesizer)


        # ==> Modulator
        modulator_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_latent if is_first else (dim_hidden + dim_latent)

            modulator_layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

            with torch.no_grad():
                # self.layers[-1][0].weight.uniform_(-1 / dim_hidden, 1 / dim_hidden)
                nn.init.kaiming_normal_(modulator_layers[-1][0].weight, a=0.0, nonlinearity='relu', mode='fan_in')

        self.modulator = modulator_layers
        # self.modulator = nn.Sequential(*modulator_layers)

        # ==> Positions
        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)


    def forward(self, latent):

        x = self.grid.clone().detach().requires_grad_()

        for i in range(self.num_layers):
            if i == 0:
                z = self.modulator[i](latent)
            else:
                z = self.modulator[i](torch.cat((latent, z), dim = -1))
            
            x = self.synthesizer[i](x)
            x = x * z
        
        x = self.synthesizer[-1](x) # shape of (h*w, 1)
        x = torch.tanh(x)
        x = x.view(-1, self.img_height, self.img_width, 1)  # reshape to (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # reshape to (batch_size, channels, height, width)
        return x
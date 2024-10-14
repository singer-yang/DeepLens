import math
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulateSiren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        dim_latent,
        num_layers,
        image_width,
        image_height,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        outermost_linear=True,
    ):
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

            synthesizer_layers.append(
                SineLayer(
                    in_features=layer_dim_in,
                    out_features=dim_hidden,
                    omega_0=layer_w0,
                    bias=use_bias,
                    is_first=is_first,
                )
            )

        if outermost_linear:
            last_layer = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                # w_std = math.sqrt(6 / dim_hidden) / w0
                # self.last_layer.weight.uniform_(- w_std, w_std)
                nn.init.kaiming_normal_(
                    last_layer.weight, a=0.0, nonlinearity="relu", mode="fan_in"
                )
        else:
            final_activation = (
                nn.Identity() if not exists(final_activation) else final_activation
            )
            last_layer = Siren(
                dim_in=dim_hidden,
                dim_out=dim_out,
                w0=w0,
                use_bias=use_bias,
                activation=final_activation,
            )
        synthesizer_layers.append(last_layer)

        self.synthesizer = synthesizer_layers
        # self.synthesizer = nn.Sequential(*synthesizer)

        # ==> Modulator
        modulator_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_latent if is_first else (dim_hidden + dim_latent)

            modulator_layers.append(
                nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU())
            )

            with torch.no_grad():
                # self.layers[-1][0].weight.uniform_(-1 / dim_hidden, 1 / dim_hidden)
                nn.init.kaiming_normal_(
                    modulator_layers[-1][0].weight,
                    a=0.0,
                    nonlinearity="relu",
                    mode="fan_in",
                )

        self.modulator = modulator_layers
        # self.modulator = nn.Sequential(*modulator_layers)

        # ==> Positions
        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(self, latent):
        x = self.grid.clone().detach().requires_grad_()

        for i in range(self.num_layers):
            if i == 0:
                z = self.modulator[i](latent)
            else:
                z = self.modulator[i](torch.cat((latent, z), dim=-1))

            x = self.synthesizer[i](x)
            x = x * z

        x = self.synthesizer[-1](x)  # shape of (h*w, 1)
        x = torch.tanh(x)
        x = x.view(
            -1, self.img_height, self.img_width, 1
        )  # reshape to (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # reshape to (batch_size, channels, height, width)
        return x


class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

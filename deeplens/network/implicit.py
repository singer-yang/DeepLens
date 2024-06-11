""" Surrogate network (implicit representation) can be used for optics/photonics optimizaiton. They have several advantages over the traditional optimization methods:

(1): differentiability. Some computations in optics are non-differentiable, e.g., the propagation operator. However, we can use the surrogate network to approximate the propagation operator, and the network is differentiable. We need to first train the network, and then freeze the network parameters, The optimizable parameters are the network input.
(2): smoothness. The surrogate network is smooth, which is good for optimization.


Models:
(1): MLP
(2): MLP + Conv
(3): Diffusion, zero123, latent diffusion

We want to use standard input and output for the network. Because the network architecture may beused for different purpose. Then for each specific application, we can post-process the output, for example, reshape the output.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from os.path import exists
from einops import rearrange


# ===============================================
# Specific network for different applications
# ===============================================
class AberNet(nn.Module):
    """ A network to represent lens aberrations.

    TODO: write the network architecture like RRDBNet.

    Hyugens wavefront.

    output convolution results.
    """
    def __init__(self):
        super(AberNet, self).__init__()

    def forward(self, x, c, wv):
        """ The input of the network is a complex field x and the curvature c.

            c should be given outside as the network should not include c. we optimize c in the main class.
            We pretrain the network and fix the model, in the main class we use the model without optimizing it.

        Args:
            x (_type_): _description_
            c (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.net(x, c, wv)
        return x


class PSPNet(nn.Module):
    """ Consider a complex field at d1 in front of a spheric surface, apply canonical transform to propagate the field to d2.

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(PSPNet, self).__init__()

        self.AmpNet = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )

        self.PhiNet = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )

    # def forward(self, amp, phi, n, wvln, c):
    #     """ 

    #     Args:
    #         u (_type_): Shape of [B, 1, H, W], complex type. 

    #     Returns:
    #         _type_: _description_
    #     """
    #     # x = torch.cat((amp, phi), dim=1)    # B, C, H, W
    #     # amp = torch.flip(u.abs(), [-1, -2])
    #     # phi = torch.flip(torch.angle(u), [-1, -2])

    #     # amp = self.AmpNet(amp)
    #     # phi = self.PhiNet(phi)
        
    #     # u = amp + 1j * phi

    #     return amp, phi

    # def CanonTrans(self, field, pupil_res=[1024, 1024], Nmax=64*64, n1=None, n2=None, normalize=False, upsample_pupil=False):
    # """ 
    #     """
    
    def forward(self, field, n, c, r, normalize=False):
        """ Naive implementation with FFT. Will be substide with PSP network in the future.

        Args:
            field (obj): Input field at d1.
            n (float): Relevant refractive index.
            c (Tensor): Curvature.
            r (float): Radius of the lens surface.
            normalize (bool): If we normalize the output. Defaults to False.

        Returns:
            _type_: _description_
        """
        if n > 1:
            n1 = 1
            n2 = n
        else:
            n1 = 1 / n
            n2 = 1

        if normalize:
            E0 = torch.sum(field.u.abs()**2)
        
        d1 = 2 / c * n1 / (n2 - n1)
        d2 = 2 / c * n2 / (n2 - n1)

        # Propagation. 
        field.prop(z=d1, n=n1)

        phi_lens = torch.exp(-1j * torch.fmod(field.k * n2 / d2 * (field.x ** 2 + field.y ** 2), 2 * np.pi))
        aper_lens = (field.x**2 + field.y**2 < r**2) * 1
        field.u = field.u * phi_lens * aper_lens
        
        field.prop(z=d2, n=n2)

        # Normalization
        if normalize:
            energy = E0 / torch.sum(field.u.abs()**2)
            field.u *= energy.sqrt()

        return field
    


# ===============================================
# Basic networks
# 1, MLP
# 2, MLPConv
# 3, Siren
# 4, ModulatedSiren
# ===============================================
    
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

        # initialize weights
        self.apply(initialize_weights)


    def forward(self, x):
        x = self.net(x)
        x = nnF.normalize(x, p=1, dim=-1)
        return x
    
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

        # Initialize weights
        self.apply(initialize_weights)

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

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

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
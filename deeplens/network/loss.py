import numpy as np
import random
from collections import deque
from math import exp
import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F



# ======================================================
# PSNR, SSIM, and VGG loss functions
# ======================================================
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - ssim(pred, target)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        vgg = vgg.to(device).eval()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:35])  # Extract features from VGG

    def forward(self, x, y):
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)
        loss = nn.functional.mse_loss(x_features, y_features)
        return loss






# ======================================================
# Other loss functions
# ======================================================
class FourierLoss(nn.Module):
    """
    # For an image reconstruction task: 
    # Blur typically manifests as a low-frequency error in an image. 
    # Noise usually consists of high-frequency fluctuations. 
    # The loss function should be able to penalize low-frequency errors more than high-frequency ones.
    """
    def __init__(self):
        super(FourierLoss, self).__init__()

    def forward(self, pred, target):
        """ Not tested. Written by GPT.
        """
        pred_fft = torch.fft(pred, 2)
        target_fft = torch.fft(target, 2)
        return torch.mean((pred_fft - target_fft) ** 2)


class AchromatLoss(nn.Module):
    def __init__(self):
        super(AchromatLoss, self).__init__()

    def forward(self, img):
        """ Reference: High-Quality Computational Imaging Through Simple Lenses. Eq. 8.
        """
        grad_x = img[:,:,:-1,1:] - img[:,:,:-1,:-1]
        grad_y = img[:,:,1:,:-1] - img[:,:,:-1,:-1]
        grad_img = grad_x + grad_y
        img = img[:,:,:-1,:-1]

        loss_rg = torch.mean(torch.abs(grad_img[:,0,:,:] * img[:,1,:,:] - grad_img[:,1,:,:] * img[:,0,:,:]))
        loss_gb = torch.mean(torch.abs(grad_img[:,1,:,:] * img[:,2,:,:] - grad_img[:,2,:,:] * img[:,1,:,:]))
        loss = loss_rg + loss_gb
        return loss


class TVLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(TVLoss, self).__init__()
        assert reduction == 'mean'

    def forward(self, pred, target):
        """ Not tested. Written by GPT.
        """
        assert len(pred.size()) == 4
        return (torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])))





# ======================================================
# PSF loss functions
# ======================================================
class PSFLoss(nn.Module):
    def __init__(self, ks):
        super(PSFLoss, self).__init__()
        self.x, self.y = torch.meshgrid(
            torch.linspace(-1, 1, ks), 
            torch.linspace(-1, 1, ks),
            indexing='xy'
        )
        self.r = torch.sqrt(self.x ** 2 + self.y ** 2) / np.sqrt(2)

    def forward(self, psf):
        """ Calculate the loss of PSF size.

        Input:
            PSF: shape [3, ks, ks]
        """
        # loss = torch.sum(psf * self.r.to(psf.device))
        r = self.r.to(psf.device)
        loss = (psf[0, ...] * r).sum()**2 + (psf[1, ...] * r).sum()**2 + (psf[2, ...] * r).sum()**2
        # loss = (psf[0, ...] * r).sum() + (psf[2, ...] * r).sum()
        return loss

class PSFRMSLoss(nn.Module):
    def __init__(self, ks):
        super(PSFRMSLoss, self).__init__()
        self.x, self.y = torch.meshgrid(
            torch.linspace(-1, 1, ks), 
            torch.linspace(-1, 1, ks),
            indexing='xy'
        )
        self.r = torch.sqrt(self.x ** 2 + self.y ** 2) / np.sqrt(2)

    def forward(self, psf):
        """ Calculate the loss of PSF centering and size.

        Input:
            PSF: shape [3, ks, ks]
        """
        device = psf.device
        psfc = [(self.x.to(device) * psf[1, ...]).mean(), (self.y.to(device) * psf[1, ...]).mean()]
        loss_center = (psfc[0].abs() + psfc[1].abs()) * psf.shape[1]

        r = torch.sqrt((self.x.to(device) - psfc[0].item()) ** 2 + (self.y.to(device) - psfc[1].item()) ** 2) / np.sqrt(2)
        loss_size = torch.sqrt((psf[0, ...] * r).sum()**2 + (psf[1, ...] * r).sum()**2 + (psf[2, ...] * r).sum()**2)

        return loss_center + loss_size

class PSFCenterLoss(nn.Module):
    def __init__(self, ks):
        super(PSFCenterLoss, self).__init__()
        self.x, self.y = torch.meshgrid(
            torch.linspace(-1, 1, ks), 
            torch.linspace(-1, 1, ks),
            indexing='xy'
        )

    def forward(self, psf):
        """ Calculate the loss of PSF center.

        Input:
            PSF: shape [3, ks, ks]
        """
        loss = (psf * self.x.to(psf.device)).sum().abs()**2 + (psf * self.y.to(psf.device)).sum().abs()**2
        return loss
    

class PSFSimLoss(nn.Module):
    def __init__(self):
        super(PSFSimLoss, self).__init__()

    def forward(self, psf):
        """ Red and Blue PSF should be similar.

        Input:
            PSF: shape [3, ks, ks]
        """
        loss = (psf[0, ...] - psf[2, ...]).abs().mean()
        return loss

class PSFDiffLoss(nn.Module):
    def __init__(self):
        super(PSFDiffLoss, self).__init__()

    def forward(self, psf):
        """ Calculate the cosine similarity of a PSF batch.

        Input:
            PSF: shape [N, 3, ks, ks]
        """
        N, C, H, W = psf.shape
        flattened_tensors = psf.view(N, -1)
        similarity_matrix = torch.mm(flattened_tensors, flattened_tensors.t())  #[N, N] similarity matrix

        loss = torch.mean(similarity_matrix)
        return loss

class PSFAlignLoss(nn.Module):
    def __init__(self, ks):
        super(PSFAlignLoss, self).__init__()
        self.x, self.y = torch.meshgrid(
            torch.linspace(-1, 1, ks), 
            torch.linspace(-1, 1, ks),
            indexing='xy')

    def calc_center(self, psf):
        """ Calculate the center of a PSF.
            
        Input:
            PSF: shape [ks, ks]
        """
        device = psf.device
        center_x = torch.sum(psf * self.x.to(device)) / torch.sum(psf)
        center_y = torch.sum(psf * self.y.to(device)) / torch.sum(psf)
        return torch.stack([center_x, center_y])

    def forward(self, psf):
        """ Calculate the loss of PSF alignment.

        Input:
            PSF: shape [3, ks, ks]
        """
        center_r = self.calc_center(psf[0, :, :])
        center_g = self.calc_center(psf[1, :, :])
        center_b = self.calc_center(psf[2, :, :])

        # loss = torch.mean((center_r - center_g) ** 2 + (center_g - center_b) ** 2)
        loss = torch.mean(center_r**2 + center_g**2 + center_b**2)
        return loss

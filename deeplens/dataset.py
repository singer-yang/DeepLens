""" 
This file contains basic dataset class, used in the AutoLens project.
"""
import torch
import math
import glob
import os
import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as nnF

from .basics import DEVICE

# ======================================
# Basic dataset class
# ======================================
class ImageDataset(Dataset):
    def __init__(self, img_dir, img_res=None):
        super(ImageDataset, self).__init__()
        self.img_paths = glob.glob(f"{img_dir}/**.png") + glob.glob(f"{img_dir}/**.jpg")
        if isinstance(img_res, int):
            img_res = [img_res, img_res]

        self.transform = transforms.Compose([
            transforms.Resize(img_res, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]) 
        img = self.transform(img)
        return img


# ======================================
# Data augmentation
# ======================================
def WaveletNoise(res=[512, 512], device=DEVICE):
    WN = torch.zeros((1, 1, res[0], res[1]), device=device)
    level = int(math.log(res[0], 2))
    for s in range(1, level+1):
        H, W = 2**s, 2**s
        N = 0.1 * (torch.rand((1, 1, H, W), device=device) - 0.5)
        LP = nnF.interpolate(nnF.avg_pool2d(N, kernel_size=(2,2)), size=(H,W))
        BP = nnF.interpolate(N - LP, (res[0], res[1]))
        WN += BP
    WN = WN.squeeze(0)

    return WN


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddSineNoise(object):
    def __init__(self, im_size=[2048, 2048], period=5, amplitude=0.1):
        self.period = period
        self.amplitude = amplitude
        self.X, self.Y = torch.meshgrid(
            torch.arange(-im_size[0]//2, im_size[0]//2, 1), 
            torch.arange(-im_size[1]//2, im_size[1]//2, 1)
        )
        
    def __call__(self, tensor):
        theta = torch.rand(1) * 2*np.pi
        return tensor + self.amplitude * torch.sin(2*np.pi/self.period*(self.X*torch.cos(theta)+self.Y*torch.sin(theta)))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddWaveletNoise(object):
    def __init__(self, im_size=(2048, 2048)):
        self.level = int(math.log(im_size[0], 2))
        self.size = im_size
        self.amplitude = 0.1


    def __call__(self, tensor):
        WN = torch.zeros_like(tensor, device=tensor.device).unsqueeze(0)

        for s in range(1, self.level+1):
            H, W = 2**s, 2**s
            N = self.amplitude * (torch.rand((1, 1, H, W), device=tensor.device) - 0.5)
            LP = nnF.interpolate(nnF.avg_pool2d(N, kernel_size=(2,2)), size=(H,W))
            BP = nnF.interpolate(N - LP, self.size)
            WN += BP

        return WN.squeeze(0) + tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddWaveletNoise(object):
    def __init__(self, im_size=(2048, 2048)):
        self.level = int(math.log(im_size[0], 2))
        self.size = im_size
        self.amplitude = 0.1

        WN = torch.zeros(im_size).unsqueeze(0).unsqueeze(0)
        for s in range(1, self.level+1):
            H, W = 2**s, 2**s
            N = self.amplitude * (torch.rand((1, 1, H, W)) - 0.5)
            LP = nnF.interpolate(nnF.avg_pool2d(N, kernel_size=(2,2)), size=(H,W))
            BP = nnF.interpolate(N - LP, self.size)
            WN += BP

        self.WN = WN.squeeze(0)


    def __call__(self, tensor):
        return self.WN + tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ==========================================================
# Functions (for generating datasets)
# ==========================================================
def gen_uniform_noise_dataset(N=200, size=[256, 256], dir='./dataset/uniform_noise'):
    os.makedirs(dir, exist_ok=True)
    for i in range(N):
        img = np.random.uniform(0, 255, (*size, 3))
        cv.imwrite(f'{dir}/{i}.png', img)


def gen_binary(N=200, size=[256, 256], dir='./dataset/binary'):
    os.makedirs(dir, exist_ok=True)
    for i in range(N):
        img = np.random.uniform(0, 1, size)
        img[img>=0.5] = 1
        img[img<0.5] = 0
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        cv.imwrite(f'{dir}/{i}.png', img*255.)


def split_integer(num, parts):
    partition = np.random.uniform(0, 1, parts)
    partition = partition/partition.sum()

    split = np.ceil(num * partition)
    split[-1] = num - np.sum(split[:-1])

    return split


def gen_checkerboard(N=200, size=[256,256], dir='./dataset/checkerboard'):
    os.makedirs(dir, exist_ok=True)
    for n in range(N):
        n0 = np.random.randint(size[0]//8, size[0])
        ls0 = split_integer(size[0], n0)
        n1 = np.random.randint(0, size[1]//2)
        ls1 = split_integer(size[1], n1)
        
        img = np.zeros((*size, 3))
        tl = [0, 0] # top-left
        br = [0, 0] # bottom-right
        for i in range(n0):
            br[0] += int(ls0[i])
            for j in range(n1):
                br[1] += int(ls1[j])
                img[tl[0]:br[0], tl[1]:br[1]] = (i+j) % 2
                tl[1] = br[1]

            tl[0] += int(ls0[i])
            tl[1] = 0
            br[1] = 0
        

        cv.imwrite(f'{dir}/{n}.png', img*255.)


def gen_sine(N=1800, size=[128, 128], dir='./dataset/sine'):
    os.makedirs(dir, exist_ok=True)
    X, Y = np.meshgrid(
        np.arange(-size[0]//2, size[0]//2, 1), 
        np.arange(-size[1]//2, size[1]//2, 1)
    )
    
    for n in range(N):
        img = np.zeros((*X.shape, 3))
        for color in range(3):
            im_gray = np.zeros_like(X).astype(np.float64)
            hybrid = np.random.randint(5)
            for _ in range(hybrid):
                wavelength = np.random.randint(size[0]//2)
                angle = np.random.rand() * np.pi
                grating = np.sin(
                    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength
                )+1 # normolize to [0, 2]
                im_gray += grating
            
            im_gray = im_gray/2/hybrid
            img[:,:,color] = im_gray

        cv.imwrite(f'{dir}/{n}.png', img*255.)


    
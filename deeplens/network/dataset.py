"""This file contains basic dataset class, used in the AutoLens project."""

import glob
import os
import zipfile

import cv2 as cv
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ======================================
# Basic dataset class
# ======================================
class ImageDataset(Dataset):
    def __init__(self, img_dir, img_res=None):
        super(ImageDataset, self).__init__()
        self.img_paths = glob.glob(f"{img_dir}/**.png") + glob.glob(f"{img_dir}/**.jpg")
        if isinstance(img_res, int):
            img_res = [img_res, img_res]

        self.transform = transforms.Compose(
            [
                transforms.AutoAugment(
                    transforms.AutoAugmentPolicy.IMAGENET,
                    transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(img_res),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        return img


class PhotographicDataset(Dataset):
    def __init__(self, img_dir, output_type="rgb", img_res=(512, 512), is_train=True):
        """Initialize the Photographic Dataset.

        Args:
            img_dir: Directory containing the images
            output_type: Type of output image format
            img_res: Image resolution. If int, creates square image of [img_res, img_res]
            is_train: Whether this is for training (with augmentation) or testing
        """
        super(PhotographicDataset, self).__init__()
        self.img_paths = glob.glob(f"{img_dir}/**.png") + glob.glob(f"{img_dir}/**.jpg")
        print(f"Found {len(self.img_paths)} images in {img_dir}")

        if isinstance(img_res, int):
            img_res = (img_res, img_res)
        self.is_train = is_train
        
        # Training transform with augmentation
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_res),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            ]
        )
        
        # Test transform without augmentation
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(img_res),
                transforms.CenterCrop(img_res),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            ]
        )
        
        self.output_type = output_type

    def __len__(self):
        return len(self.img_paths)

    def sample_iso(self):
        return torch.randint(100, 400, (1,))[0].float()

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        
        # Transform
        if self.is_train:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)

        # Random ISO value
        iso = self.sample_iso()

        return {
            "img": img,
            "iso": iso,
            "output_type": self.output_type,
        }

# ======================================
# Online dataset
# ======================================
def download_and_unzip_div2k(destination_folder):
    urls = {
        "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename, url in urls.items():
        zip_path = os.path.join(destination_folder, filename)

        # Download the dataset
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB

        with open(zip_path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
        print(f"Download of {filename} complete.")

        # Unzip the dataset
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Unzipping of {filename} complete.")

        # Remove the zip file
        os.remove(zip_path)


# ======================================
# Data augmentation
# ======================================
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddSineNoise(object):
    def __init__(self, im_size=[2048, 2048], period=5, amplitude=0.1):
        self.period = period
        self.amplitude = amplitude
        self.X, self.Y = torch.meshgrid(
            torch.arange(-im_size[0] // 2, im_size[0] // 2, 1),
            torch.arange(-im_size[1] // 2, im_size[1] // 2, 1),
        )

    def __call__(self, tensor):
        theta = torch.rand(1) * 2 * np.pi
        return tensor + self.amplitude * torch.sin(
            2
            * np.pi
            / self.period
            * (self.X * torch.cos(theta) + self.Y * torch.sin(theta))
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


# ==========================================================
# Functions (for generating datasets)
# ==========================================================
def gen_uniform_noise_dataset(N=200, size=[256, 256], dir="./dataset/uniform_noise"):
    os.makedirs(dir, exist_ok=True)
    for i in range(N):
        img = np.random.uniform(0, 255, (*size, 3))
        cv.imwrite(f"{dir}/{i}.png", img)


def gen_binary(N=200, size=[256, 256], dir="./dataset/binary"):
    os.makedirs(dir, exist_ok=True)
    for i in range(N):
        img = np.random.uniform(0, 1, size)
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        cv.imwrite(f"{dir}/{i}.png", img * 255.0)


def split_integer(num, parts):
    partition = np.random.uniform(0, 1, parts)
    partition = partition / partition.sum()

    split = np.ceil(num * partition)
    split[-1] = num - np.sum(split[:-1])

    return split


def gen_checkerboard(N=200, size=[256, 256], dir="./dataset/checkerboard"):
    os.makedirs(dir, exist_ok=True)
    for n in range(N):
        n0 = np.random.randint(size[0] // 8, size[0])
        ls0 = split_integer(size[0], n0)
        n1 = np.random.randint(0, size[1] // 2)
        ls1 = split_integer(size[1], n1)

        img = np.zeros((*size, 3))
        tl = [0, 0]  # top-left
        br = [0, 0]  # bottom-right
        for i in range(n0):
            br[0] += int(ls0[i])
            for j in range(n1):
                br[1] += int(ls1[j])
                img[tl[0] : br[0], tl[1] : br[1]] = (i + j) % 2
                tl[1] = br[1]

            tl[0] += int(ls0[i])
            tl[1] = 0
            br[1] = 0

        cv.imwrite(f"{dir}/{n}.png", img * 255.0)


def gen_sine(N=1800, size=[128, 128], dir="./dataset/sine"):
    os.makedirs(dir, exist_ok=True)
    X, Y = np.meshgrid(
        np.arange(-size[0] // 2, size[0] // 2, 1),
        np.arange(-size[1] // 2, size[1] // 2, 1),
    )

    for n in range(N):
        img = np.zeros((*X.shape, 3))
        for color in range(3):
            im_gray = np.zeros_like(X).astype(np.float64)
            hybrid = np.random.randint(5)
            for _ in range(hybrid):
                wavelength = np.random.randint(size[0] // 2)
                angle = np.random.rand() * np.pi
                grating = (
                    np.sin(
                        2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
                    )
                    + 1
                )  # normolize to [0, 2]
                im_gray += grating

            im_gray = im_gray / 2 / hybrid
            img[:, :, color] = im_gray

        cv.imwrite(f"{dir}/{n}.png", img * 255.0)

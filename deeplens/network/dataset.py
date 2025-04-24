"""Basic and common dataset classes."""

import glob
import os
import zipfile

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ======================================
# Basic dataset class
# ======================================
class ImageDataset(Dataset):
    """Basic dataset class for image data. Loads images from a directory."""

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
    """Loads images and samples ISO values from a directory. The data dict will be used for image simulation, then network training."""

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
        # print(f"Found {len(self.img_paths)} images in {img_dir}")

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
# Download datasets
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

        # Remove the zip files
        os.remove(zip_path)


def download_bsd300(destination_folder="./datasets"):
    """Download the BSDS300 dataset.

    Reference:
        [1] https://github.com/pytorch/examples/blob/main/super_resolution/data.py#L10
    """
    import tarfile
    import urllib.request
    from os import remove
    from os.path import basename, exists, join

    output_image_dir = join(destination_folder, "BSDS300/images")

    if not exists(output_image_dir):
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(destination_folder, basename(url))
        with open(file_path, "wb") as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, destination_folder)

        remove(file_path)

    return output_image_dir

"""Dataset class and download functions.

The following datasets are supported:
- BSDS300 (300 images, 22MB)
- DIV2K (800 images, 3.98GB)
- FLICK2K (2650 images, 11.6GB)
- DIV8K (1504 images, 46.3GB)
- MIT5K (5000 images, ~50GB)
"""

import glob
import os
import zipfile

import requests
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
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

    def __init__(self, img_dir, img_res=(512, 512), iso_range=(100, 400), is_train=True):
        """Initialize the Photographic Dataset.

        Args:
            img_dir: Directory containing the images
            img_res: Image resolution. If int, creates square image of [img_res, img_res]
            iso_range: ISO range. Defaults to (100, 400).
            iso_scale: ISO scale. Defaults to 1000.
            is_train: Whether this is for training (with augmentation) or testing
        """
        super(PhotographicDataset, self).__init__()
        self.img_paths = glob.glob(f"{img_dir}/**.png") + glob.glob(f"{img_dir}/**.jpg")
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"
        print(f"Found {len(self.img_paths)} images in {img_dir}")

        if isinstance(img_res, int):
            img_res = (img_res, img_res)
        self.img_res = img_res
        self.iso_range = iso_range
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

    def __len__(self):
        return len(self.img_paths)

    def sample_iso(self):
        """Sample ISO value from the ISO range."""
        iso_low, iso_high = self.iso_range
        return torch.randint(iso_low, iso_high, (1,))[0].float()

    def sample_field(self):
        """Sample field value from the field range [-1, 1] on x and y axis."""
        return torch.rand(2) * 2 - 1

    def __getitem__(self, idx):
        # Read a RGB image
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.is_train:
            # Train transform
            img = self.train_transform(img)
            iso = self.sample_iso()
            field_center = self.sample_field()
        else:
            # Test transform (we can assign fixed field value and ISO value for testing)
            img = self.test_transform(img)
            iso = self.sample_iso()
            field_center = self.sample_field()
        
        return {
            "img": img,
            "iso": iso,
            "iso_scale": 1000, # used to normalize the ISO value
            "field_center": field_center,
        }


# ======================================
# Download datasets
# ======================================
def download_bsd300(destination_folder="./datasets"):
    """Download the BSDS300 dataset (300 images, 22MB).

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

def download_div2k(destination_folder):
    """Download the DIV2K dataset (800 images, 3.98GB)."""
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

def download_flick2k(destination_folder="./datasets"):
    """Download the FLICK2K dataset (2650 images, 11.6GB).
    
    You can directly download the zip file from the following URL:
        https://huggingface.co/datasets/yangtao9009/Flickr2K/blob/main/Flickr2K.zip
    """
    # Download
    zip_path = hf_hub_download(
        repo_id="yangtao9009/Flickr2K",
        repo_type="dataset",
        filename="Flickr2K.zip",
        revision="main"  # or a specific commit/tag
    )
    print("Flickr2K downloaded to:", zip_path)

    # Unzip
    out_dir = destination_folder
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("Flickr2K extracted to:", out_dir)

def download_div8k(destination_folder="./datasets"):
    """Download the DIV8K dataset (1504 images, 46.3GB).
    
    You can directly download the zip file from the following URL:
        https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet/blob/main/DIV8K.zip
    """
    # Download
    zip_path = hf_hub_download(
        repo_id="Iceclear/DIV8K_TrainingSet",
        repo_type="dataset",
        filename="DIV8K.zip",
        revision="main"
    )
    print("DIV8K downloaded to:", zip_path)

    # Unzip
    out_dir = destination_folder
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("DIV8K extracted to:", out_dir)

def download_mit5k(destination_folder="./datasets"):
    """Download the MIT5K dataset (5000 images, ~50GB).
    
    You can directly download the zip file from the following URL:
        https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar
    """
    pass
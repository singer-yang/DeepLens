import logging
import os
import random
from glob import glob

import cv2 as cv
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import torch.nn.functional as F


# ==================================
# Image IO
# ==================================
def img2batch(img):
    """Convert image in any type to tensor batch.

    Args:
        img: image tensor (H, W, C) or (C, H, W).

    Returns:
        batch: batch tensor (1, C, H, W).
    """
    # Tensor shape
    if len(img.shape) == 2:
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
        else:
            raise ValueError("Image should be numpy array.")
    
    elif len(img.shape) == 3:
        if isinstance(img, np.ndarray):
            assert img.shape[-1] in [1, 3], "Image channel should be 1 or 3."
            img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2) # (H, W, C) -> (1, C, H, W)
        else:
            raise ValueError("Image should be numpy array.")
    
    # Tensor dtype
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0
    elif img.dtype == torch.float32:
        pass
    else:
        raise ValueError("Image type should be uint8 or float32.")
        
    return img

# ==================================
# Image batch quality evaluation
# ==================================
def batch_PSNR(img_clean, img):
    """Compute PSNR for image batch."""
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    PSNR = 0.0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Img_clean[i, :, :, :], Img[i, :, :, :])
    return round(PSNR / Img.shape[0], 4)

def batch_psnr(pred, target, max_val=1.0, eps=1e-8):  
    """Calculate PSNR between two image batches.  

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    
    Args:  
        pred (torch.Tensor): Predicted images, shape [B, C, H, W]  
        target (torch.Tensor): Target images, shape [B, C, H, W]  
        max_val (float): Maximum value of the images (1.0 for normalized, 255 for uint8)  
        eps (float): Small constant to avoid log(0)  
        
    Returns:  
        torch.Tensor: PSNR value for each image in batch, shape [B]  
    """  
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"  
    
    # Calculate MSE along spatial and channel dimensions  
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Shape: [B]  
    
    # Calculate PSNR  
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + eps))  
    
    return psnr.item()

def batch_SSIM(img, img_clean):
    """Compute SSIM for image batch."""
    return batch_ssim(img, img_clean)

def batch_ssim(img, img_clean):
    """Compute SSIM for image batch.
    
    Args:
        img (torch.Tensor): Input image batch, shape [B, C, H, W]
        img_clean (torch.Tensor): Reference image batch, shape [B, C, H, W]
        
    Returns:
        float: Average SSIM score across batch
    """
    # Convert to numpy arrays in range [0, 255]
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    
    SSIM = 0.0
    for i in range(Img.shape[0]):
        # Auto detect if multichannel based on number of dimensions
        if Img.shape[1] > 1:  # Multiple channels
            SSIM += structural_similarity(Img_clean[i, ...], Img[i, ...], channel_axis=0)
        else:  # Single channel
            SSIM += structural_similarity(Img_clean[i, 0, ...], Img[i, 0, ...])
            
    return round(SSIM / Img.shape[0], 4)

def batch_LPIPS(img, img_clean):
    """Compute LPIPS loss for image batch."""
    device = img.device
    loss_fn = lpips.LPIPS(net="vgg", spatial=True)
    loss_fn.to(device)
    dist = loss_fn.forward(img, img_clean)
    return dist.mean().item()


# ==================================
# Image batch normalization
# ==================================
def normalize_ImageNet(batch):
    """Normalize dataset by ImageNet(real scene images) distribution."""
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std
    return batch_out


def denormalize_ImageNet(batch):
    """Convert normalized images to original images to compute PSNR."""
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = batch * std + mean
    return batch_out


# ==================================
# EDoF
# ==================================
def foc_dist_balanced(d1, d2):
    """When focus to foc_dist, d1 and d2 will have the same CoC.

    Reference: https://en.wikipedia.org/wiki/Circle_of_confusion
    """
    foc_dist = 2 * d1 * d2 / (d1 + d2)
    return foc_dist


# ==================================
# AutoLens
# ==================================
def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get all .png files in the image_folder
    images = glob(os.path.join(image_folder, "*.png"))
    # images.sort()  # Sort the images by name
    images.sort(key=lambda x: os.path.getctime(x))  # Sort the images by creation time

    if not images:
        print("No PNG images found in the provided directory.")
        return

    # Read the first image to get the dimensions
    first_image = cv.imread(images[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video_writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    for image_path in tqdm(images):
        img = cv.imread(image_path)
        video_writer.write(img)

    # Release the video writer object
    video_writer.release()
    print(f"Video saved as {output_video_path}")


# ==================================
# Experimental logging
# ==================================
def gpu_init(gpu=0):
    """Initialize device and data type.

    Returns:
        device: which device to use.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_tensor_type("torch.FloatTensor")
    return device


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def set_logger(dir="./"):
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel("INFO")

    fhlr = logging.FileHandler(f"{dir}/output.log")
    fhlr.setFormatter(formatter)
    fhlr.setLevel("INFO")

    # fhlr2 = logging.FileHandler(f"{dir}/error.log")
    # fhlr2.setFormatter(formatter)
    # fhlr2.setLevel('WARNING')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    # logger.addHandler(fhlr2)

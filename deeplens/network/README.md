# Network Package

The `network` package in DeepLens provides a collection of neural network architectures, loss functions, and data handling utilities essential for training and implementing end-to-end computational imaging pipelines.

This package is a core component for tasks such as:
-   Image reconstruction and restoration.
-   Implementing surrogate models for optical elements.
-   Training neural networks for optics-vision co-design.

## Modules and Sub-packages

### `dataset.py`
This module contains PyTorch `Dataset` classes for loading image data and provides helper functions to download several standard image datasets, including:
-   BSDS300
-   DIV2K
-   FLICK2K
-   DIV8K
-   MIT5K

### `loss/`
This sub-package contains various loss functions commonly used in image processing tasks:
-   `perceptual_loss.py`: Implements perceptual loss (e.g., LPIPS) for more visually pleasing results.
-   `psnr_loss.py`: Provides Peak Signal-to-Noise Ratio (PSNR) loss.
-   `ssim_loss.py`: Implements Structural Similarity Index (SSIM) loss.

### `reconstruction/`
This sub-package includes several neural network architectures designed for image reconstruction and restoration:
-   `unet.py`: A standard U-Net implementation.
-   `nafnet.py`: Implementation of the NAFNet architecture.
-   `restormer.py`: Implementation of the Restormer model.

### `surrogate/`
This sub-package contains network models that can be used as surrogates for complex optical phenomena, enabling faster simulation and end-to-end optimization:
-   `mlp.py` & `mlpconv.py`: Implementations of Multi-Layer Perceptrons and MLP-convolutions.
-   `siren.py` & `modulate_siren.py`: Implementations of Sinusoidal Representation Networks (SIREN).
-   `psfnet_mplconv.py`: A network for point spread function (PSF) modeling.

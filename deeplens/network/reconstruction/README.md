# Reconstruction Package

The `reconstruction` package provides a collection of neural network architectures for image reconstruction and restoration tasks. These networks serve as simple examples for recovering high-quality images from degraded or distorted inputs captured through optical systems.

This package is essential for:
-   Reconstructing sharp images from optical aberrations and distortions.
-   Restoring image quality in end-to-end differentiable imaging pipelines.
-   Implementing joint optics-network optimization for computational cameras.
-   Evaluating different neural architectures for vision restoration tasks.

## Network Architectures

-   `unet.py`: UNet with residual blocks implementation by Xinge Yang, based on ResUNet-a architecture.
-   `nafnet.py`: NAFNet (Nonlinear Activation Free Network) implementation for image restoration.
-   `restormer.py`: Restormer - efficient Transformer for high-resolution image restoration.
-   `swinir.py`: SwinIR - image restoration using Swin Transformer architecture.


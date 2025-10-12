# Surrogate Package

The `surrogate` package provides neural network architectures that serve as surrogate models for complex optical phenomena. These networks serve as simple examples for enabling faster simulation and end-to-end optimization of optical systems by learning to approximate computationally expensive optical processes.

This package is essential for:
-   Modeling spatially varying Point Spread Functions (PSF) of optical systems.
-   Accelerating optical simulations during training and optimization.
-   Enabling differentiable approximations of wave propagation and diffraction.
-   Supporting joint optimization of optics and computational processing.

## Network Architectures

-   `mlp.py`: Multi-Layer Perceptron for low-resolution intensity/amplitude PSF function prediction.
-   `mlpconv.py`: MLP encoder with convolutional decoder for high-resolution PSF function prediction.
-   `siren.py`: Sinusoidal Representation Network (SIREN) for implicit neural representations.
-   `modulate_siren.py`: Modulated SIREN with adaptive frequency modulation.
-   `psfnet_mplconv.py`: MLP-Conv network architecture for spatially varying PSF representation.


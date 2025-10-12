# Optics Package

The `optics` package is the foundational module of DeepLens, providing the core functionalities for both geometric and wave optics simulations. It contains the fundamental classes for representing rays and complex wave fields, as well as the necessary tools for simulating their propagation through optical systems.

This package is essential for:
-   Defining and tracing rays through optical elements.
-   Simulating diffraction and interference effects using wave optics.
-   Modeling various optical components like surfaces and materials.
-   Calculating optical performance metrics such as Point Spread Functions (PSF).

## Key Modules

-   `basics.py`: This module defines fundamental variables, constants (e.g., wavelengths, sampling densities), and the base `DeepObj` class that provides common functionalities like device placement (`.to()`) and data type conversion (`.astype()`) for other classes in the package.

-   `ray.py`: Contains the `Ray` class, which is the cornerstone of geometric ray tracing in DeepLens. It encapsulates the properties of optical rays, including their origin, direction, wavelength, and validity.

-   `wave.py`: Implements the `ComplexWave` class for wave optics simulations. This module includes various methods for wave propagation, such as the Angular Spectrum Method (ASM) and Rayleigh-Sommerfeld diffraction, enabling the simulation of diffraction and interference.

-   `psf.py`: Provides tools for calculating the Point Spread Function (PSF) of an optical system, a critical metric for assessing image quality.

-   `materials.py`: Defines a library of optical materials and their properties, which is crucial for realistic simulations.

-   `loss.py`: Contains custom loss functions tailored for optical design optimization.

## Sub-packages

-   `geometric_surface/`: This sub-package includes classes for various types of geometric surfaces (e.g., `Aspheric`, `Conic`, `Cylindric`) that can be used to build refractive lenses.

-   `diffractive_surface/`: Contains implementations of diffractive optical elements (DOEs) and metasurfaces.

-   `geometric_phase/`: Provides tools for modeling geometric phase surfaces.

-   `material/`: Implements material properties and dispersion models (e.g., Sellmeier's equation) for accurate simulation across different wavelengths.

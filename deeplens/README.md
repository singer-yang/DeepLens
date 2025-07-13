# `deeplens` Package Structure

This document outlines the file structure of the `deeplens` package.

-   **Core Files**:
    -   `lens.py`: Base class for a lens system.
    -   `geolens.py`: Implements refractive lens systems.
    -   `diffraclens.py`: Implements diffractive lens systems.
    -   `hybridlens.py`: Implements hybrid refractive-diffractive lens systems.
    -   `camera.py`: Defines a camera system combining a lens and a sensor.
    -   `psfnet.py`: A network to learn the Point Spread Function (PSF).
    -   `utils.py`: Utility functions.

-   **`optics/`**: Contains modules for optical simulations.
    -   `geometric_surface/`: Defines various geometric surfaces for refractive lenses.
    -   `diffractive_surface/`: Defines various diffractive surfaces.
    -   `material/`: Contains material data for lenses.
    -   `ray.py`: Implements ray tracing functionalities.
    -   `wave.py`: Implements wave optics propagations.

-   **`sensor/`**: Simulates different sensor types and includes an ISP (Image Signal Processor) pipeline.
    -   `isp_modules/`: Contains various modules for the ISP pipeline like demosaicing, white balance, etc.
    -   `mono_sensor.py`, `rgb_sensor.py`, `event_sensor.py`: Implement different sensor types.

-   **`network/`**: Contains neural network models.
    -   `surrogate/`: Includes surrogate models for optical elements (e.g., SIREN, MLP).
    -   `reconstruction/`: Contains networks for image reconstruction (e.g., UNet, NAFNet).
    -   `loss/`: Defines various loss functions.

-   **`geolens_pkg/`**: A helper package for geometry-related operations, including I/O, optimization, and visualization.


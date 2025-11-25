Architecture
============

This section describes the high-level software architecture of DeepLens. Understanding this structure helps in navigating the codebase and extending it for custom computational imaging applications.

Overview
--------

The core abstraction in DeepLens is the **Camera**, which acts as a complete image simulator. A Camera is composed of two main components: a **Lens** (optical system) and a **Sensor** (electronic system).

.. code-block:: text

    Camera
    ├── Lens (Optical System)
    │   ├── GeoLens (Ray-traced refractive lens)
    │   ├── ParaxialLens (Thin lens / Circle of Confusion model)
    │   ├── DiffractiveLens (Wave-based diffractive optics)
    │   ├── HybridLens (Refractive + Diffractive combination)
    │   └── PSFNetLens (Neural network surrogate model)
    └── Sensor (Electronic System)
        ├── Noise Model (Read/Shot noise)
        └── ISP (Image Signal Processor)

Camera System
-------------

The :class:`~deeplens.camera.Camera` class is the main entry point for end-to-end simulation. It connects the optical simulation with the sensor simulation.

- **Input**: A clean, high-resolution image (representing the scene radiance).
- **Process**:
    1.  **Unprocess**: The input sRGB image is converted to linear RGB space using the invertible ISP.
    2.  **Lens Simulation**: The lens creates a degraded image on the sensor plane (optical image) based on its Point Spread Function (PSF) or ray tracing. This step simulates aberrations, blur, and distortions.
    3.  **Sensor Simulation**: The optical image is converted into digital raw data (Bayer pattern). This step adds photon shot noise, read noise, and quantization.
    4.  **ISP**: The raw data is processed back into a displayable RGB image (demosaicing, white balance, color correction, gamma, etc.).
- **Output**: A simulated "captured" image that mimics what a real camera would produce, along with ground truth for training.

Lens Module
-----------

The :class:`~deeplens.lens.Lens` class is the base class for all optical systems. It defines the common interface for PSF calculation, rendering, and analysis. DeepLens provides several specialized lens types:

*   **GeoLens**: The most accurate model for refractive lens systems. Uses differentiable ray tracing to compute PSFs and simulate optical aberrations (spherical, coma, astigmatism, distortion, chromatic). Supports multi-element lens systems with aspherical surfaces.

*   **ParaxialLens**: A simplified thin-lens model based on the ABCD matrix formalism. Models defocus (Circle of Confusion) but not higher-order aberrations. Useful for fast depth-of-field simulation and dual-pixel autofocus modeling. Commonly used in rendering software like Blender.

*   **DiffractiveLens**: Models pure diffractive optical elements (DOEs) using wave propagation (Angular Spectrum Method). Supports various diffractive surface types including Fresnel lenses, Binary DOEs, Zernike polynomials, and pixel-wise phase masks.

*   **HybridLens**: Combines refractive and diffractive optics using a differentiable ray-wave model. Rays are traced through the refractive elements, then converted to wavefronts at the DOE plane. Enables end-to-end optimization of hybrid optical systems.

*   **PSFNetLens**: A neural network surrogate model that approximates the PSF of a ``GeoLens``. Trained on ray-traced PSF data, it provides orders of magnitude faster PSF computation while remaining differentiable. Ideal for real-time applications and large-scale optimization.

Sensor Module
-------------

The :class:`~deeplens.sensor.Sensor` class handles the conversion from optical irradiance to digital signals. DeepLens provides several sensor types:

*   **Sensor**: Base sensor class with noise model and basic ISP.
*   **RGBSensor**: RGB Bayer sensor with full ISP pipeline (used by ``Camera``).
*   **MonoSensor**: Monochrome sensor without color filter array.
*   **EventSensor**: Event-based (DVS) sensor model.

Key Features
~~~~~~~~~~~~

*   **Noise Simulation**: Accurately models photon shot noise (signal-dependent) and read noise (signal-independent) based on ISO settings.

*   **ISP Pipeline**: DeepLens provides multiple ISP implementations:

    *   :class:`~deeplens.sensor.isp.SimpleISP`: Basic pipeline with essential modules.
    *   :class:`~deeplens.sensor.isp.InvertibleISP`: Differentiable and invertible pipeline, allowing researchers to "unprocess" real sRGB images into linear raw space for realistic simulation inputs.
    *   :class:`~deeplens.sensor.isp.OpenISP`: Full-featured pipeline based on fast-openISP.

*   **ISP Modules**: Modular components including black level compensation, demosaicing, auto white balance, color correction matrix, and gamma correction.

GeoLens Architecture
--------------------

The :class:`~deeplens.geolens.GeoLens` is the most commonly used class for designing refractive optics. It is designed using a **Mixin** architecture to separate concerns and keep the codebase modular.

Class Hierarchy
~~~~~~~~~~~~~~~

``GeoLens`` inherits from multiple specialized classes, each handling a specific aspect of the lens functionality:

.. code-block:: python

    class GeoLens(Lens, GeoLensEval, GeoLensOptim, GeoLensVis, GeoLensIO, GeoLensTolerance):
        ...

*   **Lens**: Base class providing common properties (device handling, basic rendering).
*   **GeoLensEval**: Contains methods for optical performance evaluation, such as Spot Diagrams, RMS error maps, Distortion grids, and MTF calculations.
*   **GeoLensOptim**: Manages differentiable optimization, including loss functions (spot size, constraints) and optimizer configuration.
*   **GeoLensVis**: Handles 2D visualization of the lens layout and ray paths.
*   **GeoLensIO**: Manages reading and writing lens files (e.g., JSON, ZEMAX formats).
*   **GeoLensTolerance**: Provides tools for tolerance analysis (Monte Carlo simulation, sensitivity analysis).

Standalone Usage
~~~~~~~~~~~~~~~~

While ``GeoLens`` can be part of a ``Camera``, it is often used **standalone** for optical design tasks that do not require sensor simulation. You can directly use a ``GeoLens`` object to:

*   Trace rays and calculate PSFs.
*   Analyze optical aberrations (coma, astigmatism, distortion).
*   Optimize lens surface parameters (curvature, thickness, aspheric coefficients) to minimize spot size or wavefront error.

This decoupling allows optical engineers to focus purely on the lens design before integrating it into a full camera system simulation.


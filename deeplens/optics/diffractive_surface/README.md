# Diffractive Surface

This folder contains definitions for various diffractive surfaces, also known as Diffractive Optical Elements (DOEs). These elements operate on the principle of wave optics, modulating the phase of an incident light wave to achieve a desired optical effect.

The core of this module is the `DiffractiveSurface` class in `diffractive.py`. It serves as the base class for all DOEs, providing a common interface and functionality. It handles the propagation of a wavefront to the surface and applies the phase modulation. Subclasses are expected to implement the `_phase_map0` method, which defines the specific phase profile of the element at a design wavelength.

## Available Surfaces

The following diffractive surfaces are available, all inheriting from the `DiffractiveSurface` base class:

-   `Binary2`: A diffractive surface with a binary phase profile (e.g., 0 and pi).
-   `Fresnel`: A diffractive lens that simulates the behavior of a Fresnel zone plate.
-   `Pixel2D`: A generic, pixelated 2D diffractive surface where the phase of each pixel can be optimized.
-   `ThinLens`: A diffractive implementation of a thin lens, focusing light based on a quadratic phase profile.
-   `Zernike`: A surface whose phase profile is described by a combination of Zernike polynomials, which are useful for representing classical optical aberrations.

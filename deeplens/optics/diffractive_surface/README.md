# Diffractive Surface

This folder contains definitions for various diffractive surfaces, also known as Diffractive Optical Elements (DOEs). These elements operate on the principle of wave optics, modulating the phase of an incident light wave to achieve a desired optical effect.

Common manufacturing methods for diffractive surfaces include:
- **Photolithography**: Standard semiconductor processing technique for creating binary or multi-level surface relief profiles.
- **Electron Beam Lithography (EBL)**: Used for high-resolution patterning, especially for creating master molds.
- **Nanoimprint Lithography (NIL)**: A cost-effective replication method suitable for mass production.
- **Single Point Diamond Turning (SPDT)**: Capable of creating diffractive structures on curved substrates (though simulation support is pending).

The core of this module is the `DiffractiveSurface` class in `diffractive.py`, which defines the common interface for all DOEs. It handles the propagation of a wavefront to the surface and applies the phase modulation.

**Note:** The current wave propagation simulation method only supports **planar** diffractive surfaces. Curved diffractive surfaces are not yet supported in this module.

## Available Surfaces

The following surfaces are available, all inheriting from the `DiffractiveSurface` base class:

-   `DiffractiveSurface`: The base class for all DOEs.
    -   `Binary2`: A diffractive surface with a binary phase profile (e.g., 0 and pi).
    -   `Fresnel`: A diffractive lens that simulates the behavior of a Fresnel zone plate.
    -   `Pixel2D`: A generic, pixelated 2D diffractive surface where the phase of each pixel can be optimized.
    -   `ThinLens`: A diffractive implementation of a thin lens, focusing light based on a quadratic phase profile.
    -   `Zernike`: A surface whose phase profile is described by a combination of Zernike polynomials, which are useful for representing classical optical aberrations.

Each of these classes implements the `_phase_map0` method, which defines the specific phase profile of the element at a design wavelength.

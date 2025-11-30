# Phase Surface

Phase surfaces are a class of diffractive surfaces consisting of a planar substrate with a diffraction pattern.

In commercial software like Zemax, diffractive surfaces are typically simulated by adding a ray bending angle to the standard refracted ray. In DeepLens, phase surfaces operate on the same principle using ray optics. (DeepLens also supports diffractive surfaces simulated via wave optics; please refer to the `deeplens/optics/diffractive_surface/` directory. Both modules represent diffractive surfaces, differing primarily in the simulation method.) Diffraction pattern can also be applied on curved surfaces, which has not been implemented yet.

Common manufacturing methods for phase surfaces include:
- **Lithography**: Standard semiconductor processing technique.
- **Nanoimprint Lithography (NIL)**: A cost-effective replication method.
- **Single Point Diamond Turning (SPDT)**: With a minimum 100nm step size.

The core of this module is the `Phase` base class in `phase.py`, which defines the common interface for all phase surfaces. It handles the ray tracing logic, coordinate transformations, and diffraction simulation.

## Available Surfaces

The following surfaces are available, all inheriting from the `Phase` base class:

-   `Phase`: The base class for all phase surfaces.
    -   `Binary2`: Represents a rotationally symmetric phase profile using even-order polynomials ($r^2, r^4, \dots$).
    -   `Cubic`: Implements a cubic phase profile using 3rd-order polynomials ($x^3, y^3, x^2y, \dots$).
    -   `Fresnel`: Simulates a Fresnel lens phase profile, defined by a focal length.
    -   `Grating`: Represents a linear diffraction grating, defined by a slope and orientation angle.
    -   `NURBS`: Uses Non-Uniform Rational B-Splines (NURBS) to define a freeform phase profile.
    -   `Poly`: A general polynomial phase surface including both even radial terms (like Binary2) and odd polynomial terms.
    -   `Quartic`: Implements a Q-type (Quartic) phase surface using 4th-order polynomial coefficients.
    -   `Zernike`: Represents the phase profile using Zernike polynomials (supports up to 37 terms).

Common real-world examples include Diffractive Optical Elements (DOEs) and metasurfaces. Canon's DO (Diffractive Optics) lenses (https://www.canon-europe.com/pro/infobank/lenses-multi-layer-diffractive-optical-element/) are a well-known application.

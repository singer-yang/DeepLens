# Geometric Surface

This folder contains the definition of various geometric surfaces used in optical systems. These surfaces can be used to build complex optical elements for ray tracing simulations.

Common manufacturing methods for geometric surfaces include:

- **Grinding and Polishing**: The traditional method for manufacturing spherical and flat surfaces.
- **Single Point Diamond Turning (SPDT)**: Used for high-precision aspheric and freeform surfaces, especially on metals and plastics.
- **Glass Molding**: Suitable for mass production of aspheric lenses.
- **Magnetorheological Finishing (MRF)**: A high-precision polishing method used to correct surface errors.

The core of this module is the `Surface` base class in `base.py`, which defines the common interface for all surfaces. It handles ray intersection, refraction, and reflection.

## Available Surfaces

The following surfaces are available, all inheriting from the `Surface` base class:

-   `Surface`: The basic surface.
    -   `Aspheric`: An aspheric surface, defined by a polynomial expansion.
    -   `AsphericNorm`: An aspheric surface with normalized polynomial parameters.
    -   `Cubic`: A surface with a cubic shape.
    -   `Spheric`: A spherical surface, a common element in lenses.
    -   `Spiral`: A surface with a spiral phase profile.
    -   `Plane`: A simple plane surface.
        -   `Aperture`: Defines an aperture stop, which limits the passage of rays.
        -   `ThinLens`: A paraxial approximation of a lens, useful for simplifying optical systems.
        -   `Mirror`: A reflective surface.



Each of these classes implements the `sag` method to define its specific shape and may have other specific parameters. They are all designed to be differentiable and can be used in gradient-based optimization of optical systems.

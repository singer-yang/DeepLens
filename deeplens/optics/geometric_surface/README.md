# Geometric Surface

This folder contains the definition of various geometric surfaces used in optical systems. These surfaces can be used to build complex optical elements for ray tracing simulations.

The core of this module is the `Surface` base class in `base.py`, which defines the common interface for all surfaces. It handles ray intersection, refraction, and reflection.

## Available Surfaces

The following surfaces are available, all inheriting from the `Surface` base class:

-   `Aperture`: Defines an aperture stop, which limits the passage of rays.
-   `Aspheric`: An aspheric surface, defined by a polynomial expansion.
-   `AsphericNorm`: An aspheric surface with normalized polynomial parameters.
-   `Cubic`: A surface with a cubic shape.
-   `Mirror`: A reflective surface.
-   `Phase`: A surface that imparts a phase shift to the incident rays, can be used to model diffractive optical elements.
-   `Plane`: A simple plane surface.
-   `Spheric`: A spherical surface, a common element in lenses.
-   `ThinLens`: A paraxial approximation of a lens, useful for simplifying optical systems.
-   `Spiral`: A surface with a spiral phase profile.

Each of these classes implements the `sag` method to define its specific shape and may have other specific parameters. They are all designed to be differentiable and can be used in gradient-based optimization of optical systems.

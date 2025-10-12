Optical Elements
================

DeepLens provides a comprehensive library of optical elements for building custom lens systems.

Geometric Surfaces
------------------

Spheric Surface
^^^^^^^^^^^^^^^

Standard spherical surface, the most common optical element.

.. code-block:: python

    from deeplens.optics import Spheric
    
    surface = Spheric(
        r=50.0,        # Radius of curvature [mm]
        d=5.0,         # Thickness to next surface [mm]
        is_square=False,  # Circular aperture
        device='cuda'
    )

**Parameters:**

* ``r``: Radius of curvature (positive for convex, negative for concave)
* ``d``: Thickness/distance to next surface
* ``is_square``: Aperture shape (True for square, False for circular)

**Mathematical Form:**

.. math::

    z = \\frac{r - \\sqrt{r^2 - x^2 - y^2}}{1}

Aspheric Surface
^^^^^^^^^^^^^^^^

Even-order aspheric surface for aberration correction.

.. code-block:: python

    from deeplens.optics import Aspheric
    
    surface = Aspheric(
        r=50.0,        # Base radius
        d=5.0,         # Thickness
        k=0.0,         # Conic constant
        ai=[0, 0, 1e-5, 0, -1e-7],  # Aspheric coefficients
        is_square=False,
        device='cuda'
    )

**Mathematical Form:**

.. math::

    z = \\frac{c\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}} + \\sum_{i=1}^{n} a_i \\rho^{2i}

where :math:`c = 1/r` is the curvature, :math:`k` is the conic constant, :math:`\\rho^2 = x^2 + y^2`.

**Conic Constant Values:**

* ``k = 0``: Sphere
* ``k = -1``: Parabola
* ``k < -1``: Hyperbola
* ``-1 < k < 0``: Ellipse
* ``k > 0``: Oblate ellipsoid

Aspheric Normalized
^^^^^^^^^^^^^^^^^^^

Normalized aspheric representation (alternative formulation).

.. code-block:: python

    from deeplens.optics import AsphericNorm
    
    surface = AsphericNorm(
        r=50.0,
        d=5.0,
        k=0.0,
        ai=[0, 0, 1e-5, 0, -1e-7],
        norm_radius=25.0,  # Normalization radius
        is_square=False,
        device='cuda'
    )

Plane Surface
^^^^^^^^^^^^^

Flat surface (infinite radius of curvature).

.. code-block:: python

    from deeplens.optics import Plane
    
    surface = Plane(
        d=10.0,  # Air gap or glass thickness
        device='cuda'
    )

Aperture Stop
^^^^^^^^^^^^^

Aperture stop defining the entrance pupil.

.. code-block:: python

    from deeplens.optics import Aperture
    
    surface = Aperture(
        r=5.0,    # Semi-diameter [mm]
        d=0.0,    # Typically zero thickness
        is_square=False,
        device='cuda'
    )

The aperture stop controls:

* F-number of the system
* Vignetting effects
* Depth of field

Thin Lens
^^^^^^^^^

Paraxial thin lens approximation.

.. code-block:: python

    from deeplens.optics import ThinLens
    
    surface = ThinLens(
        foclen=50.0,  # Focal length [mm]
        d=10.0,       # Distance to next surface
        r=10.0,       # Semi-diameter
        device='cuda'
    )

Useful for:

* Quick prototyping
* Paraxial analysis
* First-order optical design

Cubic Phase Surface
^^^^^^^^^^^^^^^^^^^

Cubic phase plate for extended depth of field.

.. code-block:: python

    from deeplens.optics import Cubic
    
    surface = Cubic(
        r=float('inf'),  # Typically flat base
        d=1.0,
        alpha=10.0,  # Cubic phase coefficient
        device='cuda'
    )

**Phase Function:**

.. math::

    \\phi(\\rho) = \\alpha (x^3 + y^3)

Phase Surface
^^^^^^^^^^^^^

General phase surface for computational optics.

.. code-block:: python

    from deeplens.optics import Phase
    
    surface = Phase(
        phase_map=torch.rand(512, 512),  # Custom phase pattern
        d=0.0,
        device='cuda'
    )

Diffractive Surfaces
--------------------

Fresnel Zone Plate
^^^^^^^^^^^^^^^^^^

Diffractive lens based on Fresnel zones.

.. code-block:: python

    from deeplens.optics.diffractive_surface import Fresnel
    
    surface = Fresnel(
        foclen=50.0,      # Focal length [mm]
        d=0.001,          # DOE thickness [mm]
        zone_num=100,     # Number of zones
        wavelength=0.550, # Design wavelength [micrometers]
        device='cuda'
    )

Binary Diffractive Surface (Binary 2-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple binary phase surface.

.. code-block:: python

    from deeplens.optics.diffractive_surface import Binary2
    
    surface = Binary2(
        phase_pattern=torch.rand(512, 512) > 0.5,  # Binary pattern
        d=0.001,
        wavelength=0.550,
        device='cuda'
    )

Pixelated Metasurface
^^^^^^^^^^^^^^^^^^^^^

High-resolution pixelated diffractive element.

.. code-block:: python

    from deeplens.optics.diffractive_surface import Pixel2D
    
    surface = Pixel2D(
        height_map=torch.rand(1024, 1024) * 0.5,  # Height in micrometers
        pixel_size=0.5,  # Pixel size in micrometers
        d=0.001,
        n_material=1.5,  # Refractive index
        wavelength=0.550,
        device='cuda'
    )

Zernike Phase Surface
^^^^^^^^^^^^^^^^^^^^^

Phase surface defined by Zernike polynomials.

.. code-block:: python

    from deeplens.optics.diffractive_surface import Zernike
    
    surface = Zernike(
        coefficients=[0, 0, 1, 0.5, 0, 0],  # Zernike coefficients
        d=0.001,
        aperture_radius=10.0,
        wavelength=0.550,
        device='cuda'
    )

**Common Zernike Terms:**

* Index 0-2: Piston, tilt
* Index 3: Defocus
* Index 4-5: Astigmatism
* Index 6-8: Coma, trefoil
* Index 9: Spherical aberration

Materials
---------

Material Database
^^^^^^^^^^^^^^^^^

DeepLens includes extensive material libraries:

.. code-block:: python

    from deeplens.optics import Material
    
    # Standard optical glass
    glass = Material('N-BK7')
    
    # Get refractive index at wavelength
    n = glass.n(wavelength=550)  # nm
    
    # Dispersion curve
    import matplotlib.pyplot as plt
    wavelengths = torch.linspace(400, 700, 100)
    indices = [glass.n(w) for w in wavelengths]
    plt.plot(wavelengths, indices)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Refractive Index')
    plt.show()

Available Catalogs
^^^^^^^^^^^^^^^^^^

* **SCHOTT**: Standard optical glasses (e.g., N-BK7, N-SF11)
* **CDGM**: Chinese optical glasses
* **PLASTIC**: Optical plastics (e.g., PMMA, PC)
* **MISC**: Special materials

Common Materials
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 30 30
   :header-rows: 1

   * - Name
     - Type
     - Refractive Index (550nm)
     - Application
   * - N-BK7
     - Crown glass
     - 1.519
     - General purpose
   * - N-SF11
     - Flint glass
     - 1.785
     - Chromatic correction
   * - PMMA
     - Plastic
     - 1.492
     - Low cost optics
   * - Fused Silica
     - Glass
     - 1.460
     - UV/IR applications

Custom Materials
^^^^^^^^^^^^^^^^

Define custom materials:

.. code-block:: python

    from deeplens.optics import Material
    
    # Sellmeier equation coefficients
    material = Material(
        name='CustomGlass',
        catalog='CUSTOM',
        sellmeier_coef=[
            1.03961212,
            0.231792344,
            1.01046945,
            0.00600069867,
            0.0200179144,
            103.560653
        ]
    )

**Sellmeier Equation:**

.. math::

    n^2 = 1 + \\frac{B_1\\lambda^2}{\\lambda^2 - C_1} + \\frac{B_2\\lambda^2}{\\lambda^2 - C_2} + \\frac{B_3\\lambda^2}{\\lambda^2 - C_3}

Ray Object
----------

The ``Ray`` class represents light rays for ray tracing.

Creating Rays
^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import Ray
    
    # Manual ray creation
    ray = Ray(
        o=torch.tensor([[0, 0, 0]]),      # Origins [mm]
        d=torch.tensor([[0, 0, 1]]),      # Directions (unit vectors)
        wavelength=0.550,                 # Wavelength [micrometers]
        device='cuda'
    )

Ray Properties
^^^^^^^^^^^^^^

.. code-block:: python

    # Ray origins
    print(ray.o.shape)  # [N, 3]
    
    # Ray directions
    print(ray.d.shape)  # [N, 3]
    
    # Ray status (1 = active, 0 = blocked)
    print(ray.ra)
    
    # Wavelength
    print(ray.wavelength)
    
    # Optical path length
    print(ray.opl)

Ray Tracing Through Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Trace ray through a surface
    ray_out = surface.ray_reaction(
        ray=ray,
        n1=1.0,      # Refractive index before
        n2=1.5,      # Refractive index after
        wavelength=0.550
    )

Wave Propagation
----------------

Angular Spectrum Method
^^^^^^^^^^^^^^^^^^^^^^^

For wave optics propagation:

.. code-block:: python

    from deeplens.optics import AngularSpectrumMethod
    
    asm = AngularSpectrumMethod(device='cuda')
    
    # Input field [H, W, 2] (complex field: real, imag)
    field_in = torch.randn(512, 512, 2).cuda()
    
    # Propagate
    field_out = asm.forward(
        field=field_in,
        distance=10.0,      # Propagation distance [mm]
        wavelength=0.550,   # Wavelength [micrometers]
        pixel_size=0.01     # Pixel size [mm]
    )

Fresnel Propagation
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import fresnel_propagation
    
    field_out = fresnel_propagation(
        field=field_in,
        distance=10.0,
        wavelength=0.550,
        pixel_size=0.01
    )

Loss Functions for Optimization
--------------------------------

DeepLens provides specialized loss functions for lens optimization:

Spot Loss
^^^^^^^^^

RMS spot size loss:

.. code-block:: python

    from deeplens.optics import SpotLoss
    
    loss_fn = SpotLoss()
    loss = loss_fn(ray_out)  # Smaller is better

RMS Wavefront Error
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import RMSLoss
    
    loss_fn = RMSLoss()
    loss = loss_fn(ray_out)

MTF Loss
^^^^^^^^

Modulation Transfer Function based loss:

.. code-block:: python

    from deeplens.optics import MTFLoss
    
    loss_fn = MTFLoss(frequency=50)  # lp/mm
    loss = loss_fn(psf)

Constraints
-----------

Physical constraints for lens optimization:

.. code-block:: python

    # Lens system constraints
    lens.init_constraints(
        min_thickness=0.5,      # Minimum edge thickness [mm]
        max_thickness=20.0,     # Maximum center thickness [mm]
        min_radius=10.0,        # Minimum radius of curvature [mm]
        max_radius=1000.0       # Maximum radius of curvature [mm]
    )
    
    # Get constraint loss
    constraint_loss = lens.loss_constraint()

Utilities
---------

Ray Sampling
^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import sample_rays
    
    # Sample rays from a point source
    rays = sample_rays(
        source='point',
        origin=[0, 0, -1000],
        pupil_radius=10.0,
        num_rays=1000,
        wavelength=0.550,
        device='cuda'
    )

Coordinate Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.utils import cart2pol, pol2cart
    
    # Cartesian to polar
    r, theta = cart2pol(x, y)
    
    # Polar to Cartesian
    x, y = pol2cart(r, theta)

Best Practices
--------------

Surface Design
^^^^^^^^^^^^^^

1. **Start Simple**: Begin with spherical surfaces, add aspherics only when needed
2. **Aperture Placement**: Place aperture stop strategically for aberration control
3. **Material Selection**: Use crown-flint pairs for achromatic designs
4. **Thickness Constraints**: Ensure physically realizable thicknesses

Optimization
^^^^^^^^^^^^

1. **Initialize Well**: Start from a reasonable design (e.g., from literature)
2. **Gradual Complexity**: Optimize spherical surfaces first, then add aspherics
3. **Multi-Wavelength**: Always optimize for multiple wavelengths
4. **Constraints**: Use physical constraints to ensure manufacturability

Performance
^^^^^^^^^^^

1. **GPU Memory**: Monitor GPU memory usage, reduce ray count if needed
2. **Batch Processing**: Process multiple field points simultaneously
3. **Precision**: Use float32 for speed, float64 for critical calculations
4. **Caching**: Cache PSFs for repeated rendering tasks

Next Steps
----------

* Learn about :doc:`sensors` for complete imaging simulation
* See :doc:`../examples/automated_lens_design` for optimization workflows
* Explore :doc:`../api/optics` for detailed API reference


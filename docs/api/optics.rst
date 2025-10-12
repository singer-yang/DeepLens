Optics API Reference
====================

This section documents the optical elements and ray tracing functionality.

Geometric Surfaces
------------------

Spheric
^^^^^^^

.. py:class:: deeplens.optics.Spheric(r, d, is_square=False, device='cuda')

   Spherical surface.

   :param r: Radius of curvature [mm]
   :param d: Thickness to next surface [mm]
   :param is_square: Square aperture (default: False for circular)
   :param device: Device

   **Attributes:**

   .. py:attribute:: r
      
      Radius of curvature (torch.nn.Parameter)

   .. py:attribute:: d
      
      Thickness (torch.nn.Parameter)

   **Methods:**

   .. py:method:: ray_reaction(ray, n1, n2, wavelength=0.55)

      Refract/reflect ray at surface.

      :param ray: Input Ray object
      :param n1: Refractive index before
      :param n2: Refractive index after
      :param wavelength: Wavelength in micrometers
      :return: Output Ray object

   .. py:method:: sag(x, y)

      Calculate surface sag.

      :param x: X coordinates [mm]
      :param y: Y coordinates [mm]
      :return: Z coordinates [mm]

   .. py:method:: surface_normal(x, y)

      Calculate surface normal.

      :param x: X coordinates
      :param y: Y coordinates
      :return: Normal vectors [N, 3]

Aspheric
^^^^^^^^

.. py:class:: deeplens.optics.Aspheric(r, d, k=0.0, ai=None, is_square=False, device='cuda')

   Even-order aspheric surface.

   :param r: Base radius [mm]
   :param d: Thickness [mm]
   :param k: Conic constant
   :param ai: Aspheric coefficients [a1, a2, ...]
   :param is_square: Square aperture
   :param device: Device

   **Mathematical Form:**
   
   .. math::

      z = \\frac{c\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}} + \\sum_{i=1}^{n} a_i \\rho^{2i}

   **Attributes:**

   .. py:attribute:: k

      Conic constant (torch.nn.Parameter)

   .. py:attribute:: ai

      Aspheric coefficients (list of torch.nn.Parameter)

Plane
^^^^^

.. py:class:: deeplens.optics.Plane(d, device='cuda')

   Flat surface.

   :param d: Thickness to next surface [mm]
   :param device: Device

Aperture
^^^^^^^^

.. py:class:: deeplens.optics.Aperture(r, d=0.0, is_square=False, device='cuda')

   Aperture stop.

   :param r: Semi-diameter [mm]
   :param d: Thickness (typically 0)
   :param is_square: Square aperture
   :param device: Device

ThinLens
^^^^^^^^

.. py:class:: deeplens.optics.ThinLens(foclen, d, r, device='cuda')

   Paraxial thin lens.

   :param foclen: Focal length [mm]
   :param d: Distance to next surface [mm]
   :param r: Semi-diameter [mm]
   :param device: Device

Diffractive Surfaces
--------------------

Fresnel
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Fresnel(foclen, d, zone_num=100, wavelength=0.550, device='cuda')

   Fresnel zone plate.

   :param foclen: Focal length [mm]
   :param d: Thickness [mm]
   :param zone_num: Number of zones
   :param wavelength: Design wavelength [μm]
   :param device: Device

Binary2
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Binary2(phase_pattern, d, wavelength=0.550, device='cuda')

   Binary phase element (2-level).

   :param phase_pattern: Binary phase pattern [H, W]
   :param d: Thickness [mm]
   :param wavelength: Design wavelength [μm]
   :param device: Device

Pixel2D
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Pixel2D(height_map, pixel_size, d, n_material=1.5, wavelength=0.550, device='cuda')

   Pixelated metasurface.

   :param height_map: Height map [H, W] in micrometers
   :param pixel_size: Pixel size [μm]
   :param d: Thickness [mm]
   :param n_material: Refractive index
   :param wavelength: Design wavelength [μm]
   :param device: Device

Zernike
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Zernike(coefficients, d, aperture_radius, wavelength=0.550, device='cuda')

   Zernike polynomial phase surface.

   :param coefficients: Zernike coefficients list
   :param d: Thickness [mm]
   :param aperture_radius: Aperture radius [mm]
   :param wavelength: Design wavelength [μm]
   :param device: Device

Ray Class
---------

.. py:class:: deeplens.optics.Ray(o, d, wavelength=0.550, device='cuda')

   Light ray representation.

   :param o: Origins [N, 3] in mm
   :param d: Directions [N, 3] (unit vectors)
   :param wavelength: Wavelength [μm]
   :param device: Device

   **Attributes:**

   .. py:attribute:: o

      Ray origins [N, 3]

   .. py:attribute:: d

      Ray directions [N, 3]

   .. py:attribute:: ra

      Ray status (1=active, 0=blocked)

   .. py:attribute:: wavelength

      Wavelength in micrometers

   .. py:attribute:: opl

      Optical path length [N]

   **Methods:**

   .. py:method:: propagate(distance)

      Propagate rays by distance.

      :param distance: Propagation distance [mm]

   .. py:method:: refract(normal, n1, n2)

      Refract rays at interface.

      :param normal: Surface normal [N, 3]
      :param n1: Refractive index before
      :param n2: Refractive index after

   .. py:method:: reflect(normal)

      Reflect rays at surface.

      :param normal: Surface normal [N, 3]

Materials
---------

.. py:class:: deeplens.optics.Material(name, catalog='SCHOTT', wave_range=[400, 700])

   Optical material with dispersion.

   :param name: Material name (e.g., 'N-BK7')
   :param catalog: Material catalog ('SCHOTT', 'CDGM', 'PLASTIC', 'MISC')
   :param wave_range: Wavelength range [min, max] in nm

   **Methods:**

   .. py:method:: n(wavelength)

      Get refractive index at wavelength.

      :param wavelength: Wavelength in nm
      :return: Refractive index (float)

   .. py:method:: abbe_number()

      Calculate Abbe number V_d.

      :return: Abbe number

   .. py:method:: dispersion()

      Get dispersion curve.

      :return: Tuple (wavelengths, indices)

**Common Materials:**

* Crown glasses: N-BK7, N-BK10, K5, etc.
* Flint glasses: N-SF11, N-SF5, F2, etc.
* Plastics: PMMA, PC, etc.

Wave Optics
-----------

AngularSpectrumMethod
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.optics.AngularSpectrumMethod(device='cuda')

   Angular spectrum propagation.

   :param device: Device

   .. py:method:: forward(field, distance, wavelength, pixel_size)

      Propagate complex field.

      :param field: Complex field [H, W, 2] (real, imag)
      :param distance: Propagation distance [mm]
      :param wavelength: Wavelength [μm]
      :param pixel_size: Pixel size [mm]
      :return: Propagated field [H, W, 2]

Fresnel Propagation
^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.optics.fresnel_propagation(field, distance, wavelength, pixel_size)

   Fresnel diffraction propagation.

   :param field: Complex field [H, W, 2]
   :param distance: Distance [mm]
   :param wavelength: Wavelength [μm]
   :param pixel_size: Pixel size [mm]
   :return: Propagated field [H, W, 2]

PSF Calculation
^^^^^^^^^^^^^^^

.. py:function:: deeplens.optics.psf.calc_psf(rays, sensor_size, sensor_res, method='ray')

   Calculate PSF from ray data.

   :param rays: Ray object at sensor
   :param sensor_size: Sensor size (W, H) [mm]
   :param sensor_res: Sensor resolution (W, H) [pixels]
   :param method: 'ray' or 'wave'
   :return: PSF tensor [C, H, W]

Loss Functions
--------------

SpotLoss
^^^^^^^^

.. py:class:: deeplens.optics.SpotLoss()

   RMS spot size loss.

   .. py:method:: forward(ray)

      Compute RMS spot loss.

      :param ray: Ray object at sensor
      :return: Loss scalar

RMSLoss
^^^^^^^

.. py:class:: deeplens.optics.RMSLoss()

   RMS wavefront error loss.

   .. py:method:: forward(ray)

      Compute RMS wavefront error.

      :param ray: Ray object
      :return: Loss scalar

MTFLoss
^^^^^^^

.. py:class:: deeplens.optics.MTFLoss(frequency=50)

   MTF-based loss.

   :param frequency: Target spatial frequency [lp/mm]

   .. py:method:: forward(psf)

      Compute MTF loss.

      :param psf: PSF tensor
      :return: Loss scalar

Utilities
---------

Coordinate Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.optics.utils.cart2pol(x, y)

   Cartesian to polar coordinates.

   :param x: X coordinates
   :param y: Y coordinates
   :return: Tuple (r, theta)

.. py:function:: deeplens.optics.utils.pol2cart(r, theta)

   Polar to Cartesian coordinates.

   :param r: Radius
   :param theta: Angle in radians
   :return: Tuple (x, y)

Sampling
^^^^^^^^

.. py:function:: deeplens.optics.utils.sample_square(N, R)

   Sample points in square.

   :param N: Number of points per side
   :param R: Half-width [mm]
   :return: Points [N*N, 2]

.. py:function:: deeplens.optics.utils.sample_circle(N, R)

   Sample points in circle.

   :param N: Approximate number of points
   :param R: Radius [mm]
   :return: Points [M, 2] where M ≈ N

Monte Carlo Integration
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.optics.monte_carlo.forward_integral(func, bounds, num_samples=10000)

   Monte Carlo integration.

   :param func: Function to integrate
   :param bounds: Integration bounds [(xmin, xmax), (ymin, ymax), ...]
   :param num_samples: Number of samples
   :return: Integral estimate

Examples
--------

Ray Tracing Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import Ray, Spheric, Material
    
    # Create surface and material
    surface = Spheric(r=50.0, d=5.0)
    material = Material('N-BK7')
    
    # Create rays
    ray = Ray(
        o=torch.tensor([[0, 0, -10]]),
        d=torch.tensor([[0, 0, 1]]),
        wavelength=0.550
    )
    
    # Trace through surface
    n1 = 1.0  # air
    n2 = material.n(550)  # BK7 at 550nm
    ray_out = surface.ray_reaction(ray, n1, n2)

Wave Propagation Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import AngularSpectrumMethod
    import torch
    
    # Create complex field
    field = torch.randn(512, 512, 2).cuda()
    
    # Propagate
    asm = AngularSpectrumMethod()
    field_out = asm.forward(
        field=field,
        distance=10.0,
        wavelength=0.550,
        pixel_size=0.01
    )

Material Database Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import Material
    import matplotlib.pyplot as plt
    import torch
    
    # Load material
    glass = Material('N-BK7')
    
    # Plot dispersion
    wavelengths = torch.linspace(400, 700, 100)
    indices = [glass.n(w.item()) for w in wavelengths]
    
    plt.plot(wavelengths, indices)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Refractive Index')
    plt.title('N-BK7 Dispersion')
    plt.show()

See Also
--------

* :doc:`../user_guide/optical_elements` - Detailed optical elements guide
* :doc:`lens` - Lens system API
* :doc:`../tutorials` - Tutorials and examples


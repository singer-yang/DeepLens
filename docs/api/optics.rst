Optics API Reference
====================

This section documents the optical elements, wave optics, ray tracing utilities, PSF utilities, and related losses.

Geometric Surfaces
------------------

Spheric
^^^^^^^

.. py:class:: deeplens.optics.geometric_surface.Spheric(c, r, d, mat2, pos_xy=[0.0, 0.0], vec_local=[0.0, 0.0, 1.0], is_square=False, device='cpu')

   Spherical surface parameterized by curvature.

   :param c: Curvature 1/R [1/mm]
   :param r: Aperture radius [mm]
   :param d: Surface z-position [mm]
   :param mat2: Material after the surface (e.g., ``'N-BK7'``, ``'air'``)
   :param pos_xy: Lateral offset [mm]
   :param vec_local: Surface local normal (unit vector)
   :param is_square: Square aperture if True, circular otherwise
   :param device: Device

   **Key methods (inherited from :class:`deeplens.optics.geometric_surface.base.Surface` unless noted):**

   - ``ray_reaction(ray, n1, n2, refraction=True)``: Intersect and refract/reflect a :class:`deeplens.optics.Ray`.
   - ``intersect(ray, n=1.0)``: Solve ray-surface intersection.
   - ``sag(x, y)``: Surface sag function z(x, y).
   - ``normal_vec(ray)``: Surface normal at intersection points.

Aspheric
^^^^^^^^

.. py:class:: deeplens.optics.geometric_surface.Aspheric(r, d, c, k, ai, mat2, pos_xy=[0.0, 0.0], vec_local=[0.0, 0.0, 1.0], is_square=False, device='cpu')

   Even-order asphere.

   :param r: Aperture radius [mm]
   :param d: Surface z-position [mm]
   :param c: Base curvature 1/R [1/mm]
   :param k: Conic constant
   :param ai: Aspheric coefficients list ``[a2, a4, a6, ...]``
   :param mat2: Material after the surface
   :param pos_xy: Lateral offset [mm]
   :param vec_local: Surface local normal (unit vector)
   :param is_square: Square aperture if True
   :param device: Device

   **Mathematical form:**

   .. math::

      z = \frac{c\rho^2}{1 + \sqrt{1-(1+k)c^2\rho^2}} + \sum_{i=1}^{n} a_{2i} \rho^{2i}

Plane
^^^^^

.. py:class:: deeplens.optics.geometric_surface.Plane(r, d, mat2, pos_xy=[0.0, 0.0], vec_local=[0.0, 0.0, 1.0], is_square=False, device='cpu')

   Flat surface (e.g., cover glass, IR filter, DOE base).

   :param r: Aperture radius [mm]
   :param d: Surface z-position [mm]
   :param mat2: Material after the surface
   :param pos_xy: Lateral offset [mm]
   :param vec_local: Surface local normal (unit vector)
   :param is_square: Square aperture if True
   :param device: Device

Aperture
^^^^^^^^

.. py:class:: deeplens.optics.geometric_surface.Aperture(r, d=0.0, pos_xy=[0.0, 0.0], vec_local=[0.0, 0.0, 1.0], is_square=False, device='cpu')

   Aperture stop.

   :param r: Semi-diameter [mm]
   :param d: Surface z-position [mm] (typically 0)
   :param pos_xy: Lateral offset [mm]
   :param vec_local: Surface local normal (unit vector)
   :param is_square: Square aperture if True
   :param device: Device

ThinLens
^^^^^^^^

.. py:class:: deeplens.optics.geometric_surface.ThinLens(r, d, f=100.0, pos_xy=[0.0, 0.0], vec_local=[0.0, 0.0, 1.0], is_square=False, device='cpu')

   Paraxial thin lens element.

   :param r: Semi-diameter [mm]
   :param d: Surface z-position [mm]
   :param f: Focal length [mm]
   :param pos_xy: Lateral offset [mm]
   :param vec_local: Surface local normal (unit vector)
   :param is_square: Square aperture if True
   :param device: Device

Diffractive Surfaces
--------------------

Fresnel
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Fresnel(d, f0=None, wvln0=0.55, res=(2000, 2000), mat='fused_silica', fab_ps=0.001, device='cpu')

   Fresnel (zone-plate-like) DOE with inverse dispersion relative to refractive lenses.

   :param d: DOE z-position [mm]
   :param f0: Design focal length at ``wvln0`` [mm]
   :param wvln0: Design wavelength [μm]
   :param res: DOE resolution (H, W) [pixels]
   :param mat: DOE material name
   :param fab_ps: Fabrication pixel size [mm]
   :param device: Device

Binary2
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Binary2(d, res=(2000, 2000), mat='fused_silica', wvln0=0.55, fab_ps=0.001, device='cpu')

   Two-level binary phase DOE with polynomial radial phase.

   :param d: DOE z-position [mm]
   :param res: DOE resolution (H, W) [pixels]
   :param mat: DOE material name
   :param wvln0: Design wavelength [μm]
   :param fab_ps: Fabrication pixel size [mm]
   :param device: Device

Pixel2D
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Pixel2D(d, phase_map_path=None, res=(2000, 2000), mat='fused_silica', wvln0=0.55, fab_ps=0.001, device='cpu')

   Pixelated metasurface with per-pixel phase parameters or a provided phase map path.

   :param d: DOE z-position [mm]
   :param phase_map_path: Optional path to a saved phase map tensor
   :param res: DOE resolution (H, W) [pixels]
   :param mat: DOE material name
   :param wvln0: Design wavelength [μm]
   :param fab_ps: Fabrication pixel size [mm]
   :param device: Device

Zernike
^^^^^^^

.. py:class:: deeplens.optics.diffractive_surface.Zernike(d, z_coeff=None, zernike_order=37, res=(2000, 2000), mat='fused_silica', fab_ps=0.001, wvln0=0.55, device='cpu')

   DOE parameterized by Zernike polynomials.

   :param d: DOE z-position [mm]
   :param z_coeff: Zernike coefficients tensor/list
   :param zernike_order: Number of Zernike terms (currently 37)
   :param res: DOE resolution (H, W) [pixels]
   :param mat: DOE material name
   :param fab_ps: Fabrication pixel size [mm]
   :param wvln0: Design wavelength [μm]
   :param device: Device

Ray Class
---------

.. py:class:: deeplens.optics.Ray(o, d, wvln=0.55, coherent=False, device='cpu')

   Light ray representation.

   :param o: Origins [..., N, 3] in mm
   :param d: Directions [..., N, 3] (unit vectors)
   :param wvln: Wavelength [μm]
   :param coherent: Enable coherent tracing and OPL accumulation
   :param device: Device

   **Attributes:**

   - ``o``: Ray origins [..., N, 3]
   - ``d``: Ray directions [..., N, 3]
   - ``valid``: Valid mask [..., N]
   - ``wvln``: Wavelength field [..., 1] in μm
   - ``opl``: Optical path length [..., 1] (coherent mode)
   - ``is_forward``: Direction flag (z-forward)

   **Selected methods:**

   - ``prop_to(z, n=1.0)``: Propagate rays to plane ``z``
   - ``centroid()``: Centroid of ray bundle
   - ``rms_error(center_ref=None)``: RMS spot size
   - ``clone(device=None)``: Deep copy optionally on a device
   - ``squeeze(dim=None)``: Squeeze dimensions

Materials
---------

.. py:class:: deeplens.optics.Material(name=None, device='cpu')

   Optical material with dispersion from built-in catalogs and custom tables.

   :param name: Material name (e.g., ``'N-BK7'``, ``'air'``) or an ``"n/V"`` string
   :param device: Device

   **Selected methods:**

   - ``refractive_index(wvln)``: Refractive index at wavelength(s) in μm
   - ``match_material(mat_table='CDGM')``: Match to closest catalog material
   - ``get_optimizer_params(lrs=[...])``: Parameters for optimizing ``n, V``

   **Notes:**

   - Supports Sellmeier, Schott, Cauchy, interpolation, and optimizable modes.

Wave Optics
-----------

ComplexWave
^^^^^^^^^^^

.. py:class:: deeplens.optics.wave.ComplexWave(u=None, wvln=0.55, z=0.0, phy_size=(4.0, 4.0), res=(2000, 2000))

   Complex scalar field with convenience constructors and propagation.

   **Selected classmethods:**

   - ``point_wave(point=(0,0,-1000.0), wvln=0.55, ...)``
   - ``plane_wave(wvln=0.55, ...)``
   - ``image_wave(img, wvln=0.55, ...)``

   **Selected methods:**

   - ``prop(prop_dist, n=1.0)``: Propagate by distance (ASM under the hood)
   - ``prop_to(z, n=1.0)``: Propagate to plane ``z``
   - ``save(filepath)`` / ``load(filepath)``
   - ``show(save_name=None, data='irr')``

Propagation Functions
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.optics.AngularSpectrumMethod(u, z, wvln, ps, n=1.0, padding=True)

   Angular spectrum propagation. ``u`` can be ``[H, W]`` or ``[B, 1, H, W]``.

.. py:function:: deeplens.optics.FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None)

   Fresnel diffraction (transfer function or impulse response form).

.. py:function:: deeplens.optics.FraunhoferDiffraction(u, z, wvln, ps, n=1.0, padding=True)

   Fraunhofer diffraction (far-field approximation).

.. py:function:: deeplens.optics.RayleighSommerfeld(u, z, wvln, ps, n=1.0, memory_saving=True)

   Rayleigh–Sommerfeld diffraction (reference, more expensive).

.. py:function:: deeplens.optics.wave.Fresnel_zmin(wvln, ps, side_length, n=1.0)

   Minimum propagation distance for Fresnel diffraction (Nyquist sampling criterion).

.. py:function:: deeplens.optics.wave.Nyquist_ASM_zmax(wvln, ps, side_length, n=1.0)

   Maximum propagation distance for Angular Spectrum Method (Nyquist sampling criterion).

PSF Utilities
-------------

.. py:function:: deeplens.optics.psf.conv_psf(img, psf)

   Convolve an image batch with one PSF per channel.

.. py:function:: deeplens.optics.psf.conv_psf_map(img, psf_map)

   Convolve with a spatial PSF map (grid-based).

.. py:function:: deeplens.optics.psf.conv_psf_map_depth_interp(img, depth, psf_map, psf_depths)

   Depth-aware PSF map interpolation and convolution.

.. py:function:: deeplens.optics.psf.conv_psf_pixel(img, psf)

   Per-pixel PSF convolution.

.. py:function:: deeplens.optics.psf.read_psf_map(filename, grid=10)

   Read a PSF map image into a tensor.

Loss Functions
--------------

PSFLoss
^^^^^^^

.. py:class:: deeplens.optics.loss.PSFLoss(w_achromatic=1.0, w_psf_size=1.0)

   Loss promoting concentrated, achromatic PSFs.

   .. py:method:: forward(psf)

      :param psf: PSF tensor [B, C, H, W] or compatible
      :return: Scalar loss

Monte Carlo Integration
-----------------------

.. py:function:: deeplens.optics.monte_carlo.forward_integral(ray, ps, ks, pointc=None, coherent=False)

   Forward Monte Carlo integration for PSF or wavefront from rays.

.. py:function:: deeplens.optics.monte_carlo.assign_points_to_pixels(points, mask, ks, x_range, y_range, interpolate=True, coherent=False, amp=None, phase=None)

   Helper for assigning sample points to pixel grids.

.. py:function:: deeplens.optics.monte_carlo.backward_integral(ray, img, ps, H, W, interpolate=True, pad=True, energy_correction=1)

   Backward integral (experimental) for ray-tracing-based rendering.

Examples
--------

Ray Tracing Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.optics import Ray, Material
    from deeplens.optics.geometric_surface import Spheric

    # Create surface and material
    surface = Spheric(c=1/50.0, r=12.5, d=5.0, mat2='N-BK7')
    glass = Material('N-BK7')

    # Create rays
    ray = Ray(
        o=torch.tensor([[0.0, 0.0, -10.0]]),
        d=torch.tensor([[0.0, 0.0, 1.0]]),
        wvln=0.55,
        device='cpu'
    )

    # Trace through surface
    n1 = 1.0  # air
    n2 = glass.refractive_index(0.55)
    ray_out = surface.ray_reaction(ray, n1, n2)

Wave Propagation Example
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.wave import ComplexWave

    # Plane wave then propagate 10 mm in air
    field = ComplexWave.plane_wave(wvln=0.55, phy_size=(5.12, 5.12), res=(512, 512))
    field.prop(10.0)

Material Dispersion Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt
    from deeplens.optics import Material

    glass = Material('N-BK7')

    w = torch.linspace(0.4, 0.7, 100)  # μm
    n = glass.refractive_index(w)

    plt.plot(w.cpu(), n.cpu())
    plt.xlabel('Wavelength [μm]')
    plt.ylabel('Refractive Index')
    plt.title('N-BK7 Dispersion')
    plt.show()

See Also
--------

* :doc:`../user_guide/optical_elements` - Detailed optical elements guide
* :doc:`lens` - Lens system API
* :doc:`../tutorials` - Tutorials and examples


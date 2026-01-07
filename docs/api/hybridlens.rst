HybridLens API Reference
========================

.. note::
   This API reference is still under development. Some details may contain mistakes or be incomplete.

``HybridLens`` models a hybrid refractive-diffractive optical system that couples a ``GeoLens`` with a diffractive optical element (DOE). It uses a differentiable ray–wave pipeline: coherent ray tracing to the DOE plane, DOE phase modulation, and wave propagation to the sensor.

Class Overview
--------------

- Inherits from ``Lens``
- Internally composes a ``GeoLens`` (refractive module) and a DOE (diffractive module)

Main Class
----------

.. py:class:: HybridLens(filename=None, device=None, dtype=torch.float64)

   Hybrid refractive–diffractive lens using a differentiable ray–wave model.

   :param filename: Path to hybrid-lens JSON file. If None, create empty hybrid lens
   :type filename: str or None
   :param device: Computing device (``'cuda'`` or ``'cpu'``). If None, auto-selects
   :type device: str or None
   :param dtype: Data type for computations (default: ``torch.float64``)
   :type dtype: torch.dtype

   .. note::
      Sensor size and resolution are read from the lens file (GeoLens section).
      If not provided, defaults of 8mm x 8mm and 2000x2000 pixels will be used.

   **Key Attributes:**

   .. py:attribute:: geolens
      :type: GeoLens

      Embedded refractive module; handles geometric ray tracing and first-order properties.

   .. py:attribute:: doe
      :type: diffractive surface (Binary2, Pixel2D, Fresnel, Zernike)

      Diffractive optical element mounted after the refractive module.

Initialization & I/O
--------------------

.. py:method:: HybridLens.read_lens_json(filename)

   Load a hybrid lens from a JSON file. Internally:

   - Builds the embedded :py:class:`GeoLens` from the same JSON
   - Instantiates DOE by ``param_model`` (``binary2``, ``pixel2d``, ``fresnel``, ``zernike``)
   - Appends a placeholder planar surface at the DOE axial position to ``geolens.surfaces``

   :param filename: Path to hybrid JSON file
   :type filename: str

.. py:method:: HybridLens.write_lens_json(lens_path)

   Save the hybrid lens to JSON. Writes refractive surfaces (excluding DOE placeholder) and DOE parameters.

   :param lens_path: Output JSON path
   :type lens_path: str

.. py:method:: HybridLens.double()

   Switch internal lens modules to double precision (``torch.float64``).

Refocusing & Scaling
--------------------

.. py:method:: HybridLens.refocus(foc_dist)

   Refocus by updating the embedded :py:meth:`GeoLens.refocus`. DOE position stays fixed relative to the refractive module.

   :param foc_dist: Target focus distance in object space (mm)
   :type foc_dist: float

.. py:method:: HybridLens.calc_scale(depth)

   Delegate to :py:meth:`GeoLens.calc_scale` to compute object-to-image scale for given depth.

   :param depth: Object depth (mm)
   :type depth: float
   :return: Scale factor (object height / image height)
   :rtype: float

Wavefield & PSF
---------------

.. py:method:: HybridLens.doe_field(point, wvln=0.589, spp=1000000)

   Compute complex wavefront at the DOE plane by coherent ray tracing through the embedded ``GeoLens``.

   :param point: Point source position ``[x, y, z]`` (normalized ``x,y`` in [-1, 1], ``z`` < 0)
   :type point: list or torch.Tensor
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param spp: Rays per point for coherent tracing (should be >= 1e6)
   :type spp: int
   :return: ``(wavefront, psf_center)`` where ``wavefront`` has shape ``[H, W]`` and ``psf_center`` is ``[x, y]`` in normalized sensor coordinates
   :rtype: (torch.Tensor, list)

.. py:method:: HybridLens.psf(points=[0.0, 0.0, -10000.0], ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT)

   Monochromatic PSF using ray–wave model:

   1. Coherent ray tracing to DOE plane to get wavefront
   2. Apply DOE phase modulation
   3. Propagate to sensor via Angular Spectrum Method (ASM)
   4. Crop around PSF center and normalize

   :param points: Point source position ``[x, y, z]`` (normalized ``x,y`` in [-1, 1], ``z`` < 0)
   :type points: list or torch.Tensor
   :param ks: Output PSF kernel size (pixels). ``None`` to return a central crop
   :type ks: int or None
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param spp: Rays per point for coherent tracing (>= 1e6 recommended)
   :type spp: int
   :return: Normalized PSF patch
   :rtype: torch.Tensor of shape ``[ks, ks]`` (or cropped ``[h, w]`` if ``ks`` is ``None``)

Visualization
-------------

.. py:method:: HybridLens.draw_layout(save_name='./DOELens.png', depth=-10000.0, ax=None, fig=None)

   Draw the hybrid system layout: refractive ray paths and illustrative wave-propagation arcs from DOE to sensor.

   :param save_name: Output figure path
   :type save_name: str
   :param depth: Object depth for ray bundles (mm)
   :type depth: float
   :param ax: Optional matplotlib axis to draw on
   :type ax: matplotlib.axes.Axes or None
   :param fig: Optional matplotlib figure
   :type fig: matplotlib.figure.Figure or None
   :return: Existing ``(ax, fig)`` if provided, otherwise saves to disk

Optimization
------------

.. py:method:: HybridLens.get_optimizer(doe_lr=1e-4, lens_lr=[1e-4, 1e-4, 1e-2, 1e-5], lr_decay=0.01)

   Construct an Adam optimizer over both refractive parameters (via :py:meth:`GeoLens.get_optimizer_params`) and DOE parameters.

   :param doe_lr: Learning rate for DOE parameters
   :type doe_lr: float
   :param lens_lr: Learning rates for refractive parameters ``[d, c, k, a]``
   :type lens_lr: list
   :param lr_decay: Decay factor for higher-order aspheric coefficients
   :type lr_decay: float
   :return: Configured optimizer
   :rtype: torch.optim.Optimizer



GeoLens API Reference
======================

.. note::
   This API reference is still under development. Some details may contain mistakes or be incomplete.

``GeoLens`` is a differentiable geometric lens simulator that uses ray tracing for modeling refractive and diffractive optical systems. It inherits from multiple mixin classes that provide specialized functionality.

Class Hierarchy
---------------

The ``GeoLens`` class inherits from:

- ``Lens`` - Base lens class
- ``GeoLensEval`` - Optical performance evaluation
- ``GeoLensOptim`` - Optimization and constraints
- ``GeoLensVis`` - 2D visualization
- ``GeoLensIO`` - File I/O operations
- ``GeoLensTolerance`` - Tolerance analysis

Main Class
----------

.. py:class:: GeoLens(filename=None, device=None, dtype=torch.float32)

   Geometric lens system using differentiable ray tracing.

   :param filename: Path to lens file (.json, .zmx, .seq). If None, creates empty lens
   :type filename: str or None
   :param device: Computing device ('cuda' or 'cpu'). If None, auto-selects
   :type device: str or None
   :param dtype: Data type for computations
   :type dtype: torch.dtype

   .. note::
      Sensor size and resolution are read from the lens file. If not provided,
      defaults of 8mm x 8mm and 2000x2000 pixels will be used.

   **Key Attributes:**

   .. py:attribute:: surfaces
      :type: list

      List of optical surfaces (Aperture, Spheric, Aspheric, etc.)

   .. py:attribute:: materials
      :type: list

      List of optical materials between surfaces

   .. py:attribute:: foclen
      :type: float

      Effective focal length in mm

   .. py:attribute:: fnum
      :type: float

      F-number (focal length / entrance pupil diameter)

   .. py:attribute:: rfov
      :type: float

      Half-diagonal field of view in radians

   .. py:attribute:: d_sensor
      :type: torch.Tensor

      Distance from first surface to sensor in mm

   .. py:attribute:: aper_idx
      :type: int

      Index of aperture stop surface

   .. py:attribute:: entr_pupilz
      :type: float

      Entrance pupil z-position in mm

   .. py:attribute:: entr_pupilr
      :type: float

      Entrance pupil radius in mm

   .. py:attribute:: exit_pupilz
      :type: float

      Exit pupil z-position in mm

   .. py:attribute:: exit_pupilr
      :type: float

      Exit pupil radius in mm

Initialization & Configuration
-------------------------------

.. py:method:: GeoLens.read_lens(filename)

   Load lens from file (.json or .zmx format).

   :param filename: Path to lens file
   :type filename: str

   .. note::
      The .seq format is not yet supported.

.. py:method:: GeoLens.post_computation()

   Compute focal length, pupil, and field of view after loading or modifying lens.
   Automatically called after ``read_lens()``.

.. py:method:: GeoLens.__call__(ray)

   Trace rays through the lens system.

   :param ray: Input rays
   :type ray: Ray
   :return: Output rays after tracing through lens
   :rtype: Ray

Ray Sampling
------------

Grid Sampling
~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_grid_rays(depth=float("inf"), num_grid=(11, 11), num_rays=16384, wvln=0.58756180, uniform_fov=True, sample_more_off_axis=False, scale_pupil=1.0)

   Sample grid rays from object space for PSF map or spot diagram analysis.

   :param depth: Object depth. ``float("inf")`` for parallel rays, finite for point sources
   :type depth: float
   :param num_grid: Grid size (cols, rows)
   :type num_grid: tuple or int
   :param num_rays: Rays per grid point
   :type num_rays: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param uniform_fov: If True, sample uniform FoV angles; otherwise sample uniform object grid
   :type uniform_fov: bool
   :param sample_more_off_axis: If True, sample more rays at off-axis fields
   :type sample_more_off_axis: bool
   :param scale_pupil: Scale factor for pupil radius
   :type scale_pupil: float
   :return: Ray object with shape [num_grid[1], num_grid[0], num_rays, 3]
   :rtype: Ray

.. py:method:: GeoLens.sample_radial_rays(num_field=5, depth=float("inf"), num_rays=2048, wvln=0.589)

   Sample radial (meridional, y-direction) rays at different field angles.

   :param num_field: Number of field angles
   :type num_field: int
   :param depth: Object depth
   :type depth: float
   :param num_rays: Rays per field
   :type num_rays: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :return: Ray object with shape [num_field, num_rays, 3]
   :rtype: Ray

Point Source Sampling
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_from_points(points=[[0.0, 0.0, -10000.0]], num_rays=2048, wvln=0.589, scale_pupil=1.0)

   Sample rays from point sources at absolute 3D coordinates.

   :param points: Point source positions in shape [3], [N, 3], or [Nx, Ny, 3]
   :type points: list or torch.Tensor
   :param num_rays: Rays per point
   :type num_rays: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param scale_pupil: Scale factor for pupil radius
   :type scale_pupil: float
   :return: Sampled rays with shape [*points.shape[:-1], num_rays, 3]
   :rtype: Ray

Parallel & Angular Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512, wvln=0.589, entrance_pupil=True, depth=-1.0, scale_pupil=1.0)

   Sample parallel rays at given field angles.

   :param fov_x: Field angle(s) in x direction in degrees
   :type fov_x: float or list
   :param fov_y: Field angle(s) in y direction in degrees
   :type fov_y: float or list
   :param num_rays: Rays per field point
   :type num_rays: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param entrance_pupil: If True, sample on entrance pupil; else on first surface
   :type entrance_pupil: bool
   :param depth: Propagation depth in z
   :type depth: float
   :param scale_pupil: Scale factor for pupil radius
   :type scale_pupil: float
   :return: Ray object with shape [len(fov_y), len(fov_x), num_rays, 3]
   :rtype: Ray

.. py:method:: GeoLens.sample_point_source(fov_x=[0.0], fov_y=[0.0], depth=-10000.0, num_rays=2048, wvln=0.589, entrance_pupil=True, scale_pupil=1.0)

   Sample point source rays at given field angles and depth.

   :param fov_x: Field angle(s) in x direction in degrees
   :type fov_x: float or list
   :param fov_y: Field angle(s) in y direction in degrees
   :type fov_y: float or list
   :param depth: Object depth in mm
   :type depth: float
   :param num_rays: Rays per field point
   :type num_rays: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param entrance_pupil: If True, use entrance pupil
   :type entrance_pupil: bool
   :param scale_pupil: Scale factor for pupil radius
   :type scale_pupil: float
   :return: Ray object with shape [len(fov_y), len(fov_x), num_rays, 3]
   :rtype: Ray

Sensor Sampling
~~~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_sensor(spp=64, wvln=0.589, sub_pixel=False)

   Sample backward rays from sensor pixels for ray-tracing rendering.

   :param spp: Samples per pixel
   :type spp: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param sub_pixel: If True, enable sub-pixel sampling
   :type sub_pixel: bool
   :return: Ray object with shape [H, W, spp, 3]
   :rtype: Ray

Helper Methods
~~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_circle(r, z, shape=[16, 16, 512])

   Sample points uniformly inside a circle at depth z.

   :param r: Circle radius
   :type r: float
   :param z: Z-coordinate
   :type z: float
   :param shape: Output shape
   :type shape: list
   :return: Sampled points with shape [*shape, 3]
   :rtype: torch.Tensor

.. py:method:: GeoLens.sample_ring_arm_rays(num_ring=8, num_arm=8, spp=2048, depth=-10000.0, wvln=0.589, scale_pupil=1.0, sample_more_off_axis=True)

   Sample rays using ring-arm pattern for optimization (from ``GeoLensOptim``).

   :param num_ring: Number of concentric rings
   :type num_ring: int
   :param num_arm: Number of radial arms
   :type num_arm: int
   :param spp: Samples per pixel
   :type spp: int
   :param depth: Object depth
   :type depth: float
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param scale_pupil: Pupil scale factor
   :type scale_pupil: float
   :param sample_more_off_axis: Sample more off-axis rays
   :type sample_more_off_axis: bool
   :return: Sampled rays
   :rtype: Ray

Ray Tracing
-----------

.. py:method:: GeoLens.trace(ray, surf_range=None, record=False)

   Trace rays through the lens (forward or backward determined automatically).

   :param ray: Input rays
   :type ray: Ray
   :param surf_range: Surface index range to trace through. If None, traces all surfaces
   :type surf_range: range or None
   :param record: If True, record intersection points at each surface
   :type record: bool
   :return: Output rays and optional intersection records
   :rtype: tuple(Ray, list or None)

.. py:method:: GeoLens.trace2sensor(ray, record=False)

   Trace rays to sensor plane.

   :param ray: Input rays
   :type ray: Ray
   :param record: If True, record intersection points
   :type record: bool
   :return: Rays at sensor plane (and optional records if record=True)
   :rtype: Ray or tuple(Ray, list)

.. py:method:: GeoLens.trace2obj(ray)

   Trace rays to object space (backward tracing).

   :param ray: Input rays from sensor
   :type ray: Ray
   :return: Output rays in object space
   :rtype: Ray

.. py:method:: GeoLens.forward_tracing(ray, surf_range, record)

   Forward ray tracing implementation.

   :param ray: Input rays
   :type ray: Ray
   :param surf_range: Surface range
   :type surf_range: range
   :param record: Record intersections
   :type record: bool
   :return: Output rays and records
   :rtype: tuple(Ray, list or None)

.. py:method:: GeoLens.backward_tracing(ray, surf_range, record)

   Backward ray tracing implementation.

   :param ray: Input rays
   :type ray: Ray
   :param surf_range: Surface range
   :type surf_range: range
   :param record: Record intersections
   :type record: bool
   :return: Output rays and records
   :rtype: tuple(Ray, list or None)

Image Rendering
---------------

Main Rendering
~~~~~~~~~~~~~~

.. py:method:: GeoLens.render(img_obj, depth=-10000.0, method="ray_tracing", **kwargs)

   Differentiable image simulation through the lens.

   :param img_obj: Input image tensor
   :type img_obj: torch.Tensor of shape [B, C, H, W]
   :param depth: Object depth in mm
   :type depth: float
   :param method: Rendering method - "ray_tracing", "psf_map", or "psf_patch"
   :type method: str
   :param kwargs: Additional method-specific arguments:
      - For "psf_map": psf_grid=(10,10), psf_ks=64
      - For "psf_patch": psf_center=(0.0,0.0), psf_ks=64
      - For "ray_tracing": spp=64
   :return: Rendered image tensor [B, C, H, W]
   :rtype: torch.Tensor

Ray Tracing Rendering
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.render_raytracing(img, depth=-10000.0, spp=64, vignetting=False)

   Render RGB image using ray tracing.

   :param img: RGB image tensor [N, 3, H, W]
   :type img: torch.Tensor
   :param depth: Object depth
   :type depth: float
   :param spp: Samples per pixel
   :type spp: int
   :param vignetting: Consider vignetting effect
   :type vignetting: bool
   :return: Rendered image [N, 3, H, W]
   :rtype: torch.Tensor

.. py:method:: GeoLens.render_raytracing_mono(img, wvln, depth=-10000.0, spp=64, vignetting=False)

   Render monochrome image using ray tracing.

   :param img: Monochrome image [N, H, W] or [N, 1, H, W]
   :type img: torch.Tensor
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param depth: Object depth
   :type depth: float
   :param spp: Samples per pixel
   :type spp: int
   :param vignetting: Consider vignetting effect
   :type vignetting: bool
   :return: Rendered image
   :rtype: torch.Tensor

.. py:method:: GeoLens.render_compute_image(img, depth, scale, ray, vignetting=False)

   Core rendering computation using bilinear interpolation.

   :param img: Image tensor [N, C, H, W] or [N, H, W]
   :type img: torch.Tensor
   :param depth: Object depth
   :type depth: float
   :param scale: Scale factor
   :type scale: float
   :param ray: Traced rays [H, W, spp, 3]
   :type ray: Ray
   :param vignetting: Consider vignetting
   :type vignetting: bool
   :return: Rendered image
   :rtype: torch.Tensor

Post-Processing
~~~~~~~~~~~~~~~

.. py:method:: GeoLens.unwarp(img, depth=-10000.0, num_grid=128, crop=True, flip=True)

   Unwarp rendered images to correct distortion.

   :param img: Rendered image [N, C, H, W]
   :type img: torch.Tensor
   :param depth: Object depth
   :type depth: float
   :param num_grid: Distortion grid resolution
   :type num_grid: int
   :param crop: Crop the unwarped image
   :type crop: bool
   :param flip: Flip vertically
   :type flip: bool
   :return: Unwarped image [N, C, H, W]
   :rtype: torch.Tensor

.. py:method:: GeoLens.analysis_rendering(img_org, save_name=None, depth=-10000.0, spp=64, unwarp=False, noise=0.0, method="ray_tracing", show=False)

   Render image and compute PSNR/SSIM for analysis.

   :param img_org: Original image [H, W, 3]
   :type img_org: numpy.ndarray or torch.Tensor
   :param save_name: Save filename prefix
   :type save_name: str or None
   :param depth: Object depth
   :type depth: float
   :param spp: Samples per pixel
   :type spp: int
   :param unwarp: Apply distortion correction
   :type unwarp: bool
   :param noise: Gaussian noise level
   :type noise: float
   :param method: Rendering method
   :type method: str
   :param show: Display result
   :type show: bool
   :return: Rendered image
   :rtype: torch.Tensor

PSF Calculation
---------------

.. py:method:: GeoLens.psf(points, ks=64, wvln=0.589, spp=None, recenter=True, model="geometric")

   Calculate Point Spread Function (PSF) using different models.

   :param points: Point source positions in shape [3], [N, 3], or [Nx, Ny, 3]
   :type points: torch.Tensor or list
   :param ks: PSF kernel size
   :type ks: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param spp: Samples per pixel (defaults to spp_psf for geometric, spp_coherent for coherent/huygens)
   :type spp: int or None
   :param recenter: Recenter PSF using chief ray
   :type recenter: bool
   :param model: PSF model - "geometric", "coherent", or "huygens"
   :type model: str
   :return: PSF tensor [ks, ks] or [N, ks, ks]
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_geometric(points, ks=64, wvln=0.589, spp=2048, recenter=True)

   Calculate incoherent geometric PSF using ray tracing.

   :param points: Point positions [N, 3]
   :type points: torch.Tensor or list
   :param ks: PSF kernel size
   :type ks: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param spp: Samples per pixel
   :type spp: int
   :param recenter: Recenter PSF using chief ray
   :type recenter: bool
   :return: PSF tensor
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_coherent(points, ks=64, wvln=0.589, spp=1000000, recenter=True)

   Calculate coherent PSF by propagating pupil field to sensor (Ray-Wave model). Alias for ``psf_pupil_prop``.

.. py:method:: GeoLens.psf_pupil_prop(points, ks=64, wvln=0.589, spp=1000000, recenter=True)

   Calculate coherent PSF by propagating pupil field to sensor using ASM.

   :param points: Point source positions
   :param ks: Kernel size
   :param wvln: Wavelength
   :param spp: Sample rays (typically 1M)
   :param recenter: Recenter PSF
   :return: PSF patch
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_huygens(points, ks=64, wvln=0.589, spp=1000000, recenter=True)

   Calculate Huygens PSF by treating every exit-pupil ray as a secondary spherical wave source.

   :param points: Point source position
   :param ks: Kernel size
   :param wvln: Wavelength
   :param spp: Sample rays
   :param recenter: Recenter PSF
   :return: Huygens PSF patch
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_map(depth=-10000.0, grid=(7, 7), ks=64, spp=2048, wvln=0.589, recenter=True)

   Calculate PSF map at different field positions.

   :param depth: Object depth
   :type depth: float
   :param grid: Grid size (grid_w, grid_h)
   :type grid: tuple or int
   :param ks: Kernel size
   :type ks: int
   :param spp: Samples per pixel
   :type spp: int
   :param wvln: Wavelength
   :type wvln: float
   :param recenter: Recenter PSFs
   :type recenter: bool
   :return: PSF map [grid_h, grid_w, 1, ks, ks]
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_center(points, method="chief_ray")

   Calculate PSF center position on sensor.

   :param points: Point source positions [..., 3]
   :type points: torch.Tensor
   :return: PSF centers [..., 2]
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf_coherent(points, ks=64, wvln=0.589, spp=1000000, recenter=True)

   Calculate coherent PSF using ray-wave model. Alias for ``psf_pupil_prop``.

   :param points: Point source position [3] or [1, 3]
   :type points: torch.Tensor or list
   :param ks: Kernel size
   :type ks: int
   :param wvln: Wavelength
   :type wvln: float
   :param spp: Sample rays (>= 1M recommended)
   :type spp: int
   :param recenter: Recenter PSF using chief ray
   :type recenter: bool
   :return: PSF patch [ks, ks]
   :rtype: torch.Tensor

.. py:method:: GeoLens.pupil_field(points, wvln=0.589, spp=1000000, recenter=True)

   Calculate complex wavefront at exit pupil using coherent ray tracing. Only single-point input is supported.

   :param points: Point source position [3] or [1, 3]
   :type points: torch.Tensor or list
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param spp: Samples (>= 1M required)
   :type spp: int
   :param recenter: Recenter PSF using chief ray
   :type recenter: bool
   :return: Tuple of (wavefront field, PSF center)
   :rtype: tuple(torch.Tensor, list)

Optical Analysis (GeoLensEval)
-------------------------------

Spot Diagrams
~~~~~~~~~~~~~

.. py:method:: GeoLens.draw_spot_radial(save_name='./lens_spot_radial.png', num_fov=5, depth=float("inf"), num_rays=16384, wvln_list=[0.656, 0.588, 0.486], show=False)

   Draw spot diagrams along meridional direction.

   :param save_name: Save filename
   :type save_name: str
   :param num_fov: Number of field of view angles
   :type num_fov: int
   :param depth: Object depth
   :type depth: float
   :param num_rays: Number of rays to sample
   :type num_rays: int
   :param wvln_list: List of wavelengths to render (RGB)
   :type wvln_list: list
   :param show: Display plot
   :type show: bool

.. py:method:: GeoLens.draw_spot_map(save_name='./lens_spot_map.png', num_grid=5, depth=-20000.0, num_rays=16384, wvln_list=[0.656, 0.588, 0.486], show=False)

   Draw spot diagram grid.

   :param save_name: Save filename
   :type save_name: str
   :param num_grid: Grid size
   :type num_grid: int
   :param depth: Object depth
   :type depth: float
   :param num_rays: Number of rays to sample
   :type num_rays: int
   :param wvln_list: List of wavelengths to render (RGB)
   :type wvln_list: list
   :param show: Display plot
   :type show: bool

.. py:method:: GeoLens.analysis_spot(num_field=3, depth=float("inf"))

   Compute RMS spot size and geometric radius.

   :param num_field: Number of fields to analyze
   :type num_field: int
   :param depth: Object depth
   :type depth: float
   :return: Dictionary with RMS and radius for each field
   :rtype: dict

RMS Error Maps
~~~~~~~~~~~~~~

.. py:method:: GeoLens.rms_map_rgb(num_grid=32, depth=-10000.0)

   Calculate RGB RMS spot error map.

   :param num_grid: Grid resolution
   :type num_grid: int
   :param depth: Object depth
   :type depth: float
   :return: RMS error map for each RGB channel
   :rtype: torch.Tensor

.. py:method:: GeoLens.rms_map(num_grid=32, depth=-10000.0, wvln=0.589)

   Calculate RMS spot error map for single wavelength.

   :param num_grid: Grid resolution
   :type num_grid: int
   :param depth: Object depth
   :type depth: float
   :param wvln: Wavelength
   :type wvln: float
   :return: RMS error map
   :rtype: torch.Tensor

Distortion
~~~~~~~~~~

.. py:method:: GeoLens.calc_distortion_2D(rfov, wvln=0.58756180, plane='meridional', ray_aiming=True)

   Calculate distortion at a specific field angle.

   :param rfov: View angle in degrees
   :type rfov: float or torch.Tensor
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param plane: 'meridional' or 'sagittal'
   :type plane: str
   :param ray_aiming: Whether the chief ray passes through the center of the stop
   :type ray_aiming: bool
   :return: Distortion at the specific field angle
   :rtype: float or numpy.ndarray

.. py:method:: GeoLens.draw_distortion_radial(rfov, save_name=None, num_points=21, wvln=0.58756180, plane='meridional', ray_aiming=True, show=False)

   Draw distortion curve vs field angle (Zemax-style).

   :param rfov: Maximum view angle in degrees
   :type rfov: float
   :param save_name: Save filename
   :type save_name: str or None
   :param num_points: Number of field samples
   :type num_points: int
   :param wvln: Wavelength in micrometers
   :type wvln: float
   :param plane: 'meridional' or 'sagittal'
   :type plane: str
   :param ray_aiming: Whether to use ray aiming
   :type ray_aiming: bool
   :param show: Display plot
   :type show: bool

.. py:method:: GeoLens.distortion_map(num_grid=16, depth=-10000.0)

   Compute distortion map for grid_sample.

   :param num_grid: Grid resolution
   :type num_grid: int
   :param depth: Object depth
   :type depth: float
   :return: Distortion map [num_grid, num_grid, 2]
   :rtype: torch.Tensor

.. py:method:: GeoLens.draw_distortion(filename=None, num_grid=16, depth=-10000.0)

   Visualize distortion map.

   :param filename: Save filename
   :type filename: str or None
   :param num_grid: Grid resolution
   :type num_grid: int
   :param depth: Object depth
   :type depth: float

MTF Analysis
~~~~~~~~~~~~

.. py:method:: GeoLens.mtf(fov, wvln=0.589)

   Calculate Modulation Transfer Function at field of view.

   :param fov: Field position [x, y] normalized in [-1, 1]
   :type fov: list or torch.Tensor
   :param wvln: Wavelength
   :type wvln: float
   :return: MTF curve
   :rtype: torch.Tensor

.. py:method:: GeoLens.psf2mtf(psf, pixel_size)
   :staticmethod:

   Convert PSF to MTF.

   :param psf: PSF tensor
   :type psf: torch.Tensor
   :param pixel_size: Pixel size in mm
   :type pixel_size: float
   :return: MTF curve
   :rtype: torch.Tensor

.. py:method:: GeoLens.draw_mtf(save_name='./lens_mtf.png', relative_fov_list=[0.0, 0.7, 1.0], depth_list=[-20000.0], psf_ks=128, show=False)

   Draw MTF curves for multiple depths, FOVs and RGB wavelengths.

   :param save_name: Save filename
   :type save_name: str
   :param relative_fov_list: List of relative field of view values (0.0 to 1.0)
   :type relative_fov_list: list
   :param depth_list: List of object depths in mm
   :type depth_list: list
   :param psf_ks: Kernel size for PSF calculation
   :type psf_ks: int
   :param show: Display plot
   :type show: bool

Vignetting
~~~~~~~~~~

.. py:method:: GeoLens.vignetting(depth=-10000.0, num_grid=64)

   Compute vignetting map.

   :param depth: Object depth
   :type depth: float
   :param num_grid: Grid resolution
   :type num_grid: int
   :return: Vignetting map
   :rtype: torch.Tensor

.. py:method:: GeoLens.draw_vignetting(filename=None, depth=-10000.0, resolution=512)

   Visualize vignetting effect.

   :param filename: Save filename
   :type filename: str or None
   :param depth: Object depth
   :type depth: float
   :param resolution: Output resolution
   :type resolution: int

Chief Ray
~~~~~~~~~

.. py:method:: GeoLens.calc_chief_ray(fov, plane="sagittal")

   Calculate chief ray for given field angle.

   :param fov: Field of view in degrees
   :type fov: float
   :param plane: "sagittal" or "meridional"
   :type plane: str
   :return: Chief ray
   :rtype: Ray

.. py:method:: GeoLens.calc_chief_ray_infinite(fov, plane="sagittal")

   Calculate chief ray for infinite object distance.

   :param fov: Field of view in degrees
   :type fov: float
   :param plane: "sagittal" or "meridional"
   :type plane: str
   :return: Chief ray
   :rtype: Ray

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.analysis(save_name="./lens", depth=float("inf"), render=False, render_unwarp=False, lens_title=None, show=False)

   Comprehensive lens analysis including layout, spot, MTF, and optional rendering.

   :param save_name: Filename prefix for outputs
   :type save_name: str
   :param depth: Object depth
   :type depth: float
   :param render: Render test image
   :type render: bool
   :param render_unwarp: Apply distortion correction
   :type render_unwarp: bool
   :param lens_title: Title for plots
   :type lens_title: str or None
   :param show: Display plots
   :type show: bool

Geometric Calculations
----------------------

Focal Length & Pupils
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.calc_foclen()

   Compute effective focal length by tracing paraxial chief ray.

.. py:method:: GeoLens.calc_pupil()

   Compute entrance and exit pupil positions and radii.

.. py:method:: GeoLens.get_entrance_pupil(paraxial=False)

   Get entrance pupil location and radius.

   :param paraxial: Use paraxial approximation
   :type paraxial: bool
   :return: (z_position, radius)
   :rtype: tuple(float, float)

.. py:method:: GeoLens.get_exit_pupil(paraxial=False)

   Get exit pupil location and radius.

   :param paraxial: Use paraxial approximation
   :type paraxial: bool
   :return: (z_position, radius)
   :rtype: tuple(float, float)

.. py:method:: GeoLens.calc_entrance_pupil(paraxial=False)

   Calculate entrance pupil by backward ray tracing from aperture.

   :param paraxial: Use paraxial mode for stability
   :type paraxial: bool
   :return: (z_position, radius)
   :rtype: tuple(float, float)

.. py:method:: GeoLens.calc_exit_pupil(paraxial=False)

   Calculate exit pupil by forward ray tracing from aperture.

   :param paraxial: Use paraxial mode for stability
   :type paraxial: bool
   :return: (z_position, radius)
   :rtype: tuple(float, float)

Field of View
~~~~~~~~~~~~~

.. py:method:: GeoLens.calc_fov()

   Compute field of view using perspective projection and ray tracing.

.. py:method:: GeoLens.set_fov(rfov)

   Set target half-diagonal field of view.

   :param rfov: Half diagonal FOV in radians
   :type rfov: float

Focal & Sensor Planes
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.calc_focal_plane(wvln=0.589)

   Calculate focus distance in object space by backward tracing.

   :param wvln: Wavelength
   :type wvln: float
   :return: Focal plane z-position
   :rtype: float

.. py:method:: GeoLens.calc_sensor_plane(depth=float("inf"))

   Calculate in-focus sensor position for given object depth.

   :param depth: Object depth
   :type depth: float
   :return: Sensor z-position
   :rtype: torch.Tensor

.. py:method:: GeoLens.calc_scale(depth)

   Calculate magnification scale (obj_height / img_height).

   :param depth: Object depth
   :type depth: float
   :return: Scale factor
   :rtype: float

Helper Methods
~~~~~~~~~~~~~~

.. py:method:: GeoLens.compute_intersection_points_2d(origins, directions)
   :staticmethod:

   Compute intersection points of 2D lines.

   :param origins: Line origins [N, 2]
   :type origins: torch.Tensor
   :param directions: Line directions [N, 2]
   :type directions: torch.Tensor
   :return: Intersection points [N*(N-1)/2, 2]
   :rtype: torch.Tensor

.. py:method:: GeoLens.find_diff_surf()

   Get list of differentiable/optimizable surface indices.

   :return: Range of optimizable surface indices
   :rtype: range

Lens Operations
---------------

Focusing & Aperture
~~~~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.refocus(foc_dist=float("inf"))

   Refocus lens by adjusting sensor position.

   :param foc_dist: Target focal distance in object space
   :type foc_dist: float

.. py:method:: GeoLens.set_fnum(fnum)

   Set F-number by adjusting aperture radius using binary search.

   :param fnum: Target F-number
   :type fnum: float

.. py:method:: GeoLens.set_target_fov_fnum(rfov, fnum)

   Set both FoV and F-number design targets.

   :param rfov: Half diagonal FOV in radians
   :type rfov: float
   :param fnum: F-number
   :type fnum: float

Shape Correction
~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.prune_surf(expand_factor=None)

   Prune surface radii to allow valid rays with expansion margin.

   :param expand_factor: Expansion factor (auto-selected if None)
   :type expand_factor: float or None

.. py:method:: GeoLens.correct_shape(expand_factor=None)

   Correct wrong lens shape during design.

   :param expand_factor: Expansion factor for pruning
   :type expand_factor: float or None
   :return: True if shape was changed
   :rtype: bool

Materials
~~~~~~~~~

.. py:method:: GeoLens.match_materials(mat_table="CDGM")

   Match all materials to closest in catalog.

   :param mat_table: Material catalog name
   :type mat_table: str

Optimization (GeoLensOptim)
---------------------------

Constraints
~~~~~~~~~~~

.. py:method:: GeoLens.init_constraints(constraint_params=None)

   Initialize lens design constraints (edge thickness, etc.).

   :param constraint_params: Custom constraint parameters
   :type constraint_params: dict or None

Loss Functions
~~~~~~~~~~~~~~

.. py:method:: GeoLens.loss_reg(w_focus=10.0, w_ray_angle=2.0, w_intersec=1.0, w_gap=0.1, w_surf=1.0)

   Empirical regularization loss for lens design.

   :param w_focus: Weight for in-focus loss
   :type w_focus: float
   :param w_ray_angle: Weight for chief ray angle
   :type w_ray_angle: float
   :param w_intersec: Weight for surface intersection
   :type w_intersec: float
   :param w_gap: Weight for air gap constraints
   :type w_gap: float
   :param w_surf: Weight for surface shape
   :type w_surf: float
   :return: Total regularization loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_infocus(target=0.005)

   Loss for focusing parallel rays.

   :param target: Target RMS error in mm
   :type target: float
   :return: Focus loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_surface()

   Penalize problematic surface shapes (sag, gradient, curvature).

   :return: Surface shape loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_intersec()

   Penalize surface self-intersection violations.

   :return: Intersection loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_gap()

   Penalize excessive air gaps and element thickness.

   :return: Gap constraint loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_ray_angle()

   Penalize large chief ray angles at sensor.

   :return: Ray angle loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_mat()

   Penalize invalid material properties.

   :return: Material loss
   :rtype: torch.Tensor

.. py:method:: GeoLens.loss_rms(depth=float("inf"), grid=None, spp=None, wvlns=None, weights=None)

   Compute RGB spot RMS loss.

   :param depth: Object depth
   :type depth: float
   :param grid: Field grid size
   :type grid: tuple or None
   :param spp: Samples per pixel
   :type spp: int or None
   :param wvlns: Wavelengths list
   :type wvlns: list or None
   :param weights: Field weights [center, 0.7, edge]
   :type weights: list or None
   :return: RMS loss
   :rtype: torch.Tensor

Optimization
~~~~~~~~~~~~

.. py:method:: GeoLens.optimize(iterations=1000, lrs=[1e-4, 1e-4, 1e-2, 1e-4], loss_type='rms', test_per_iter=100, decay=0.01, save_dir='./results', start_epoch=0)

   Optimize lens design to minimize RMS errors.

   :param iterations: Number of optimization iterations
   :type iterations: int
   :param lrs: Learning rates [d, c, k, a] for surface parameters
   :type lrs: list
   :param loss_type: Loss type - 'rms', 'infocus', etc.
   :type loss_type: str
   :param test_per_iter: Test every N iterations
   :type test_per_iter: int
   :param decay: Learning rate decay for higher-order terms
   :type decay: float
   :param save_dir: Directory for saving results
   :type save_dir: str
   :param start_epoch: Starting epoch number
   :type start_epoch: int
   :return: Loss history
   :rtype: list

.. py:method:: GeoLens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.01, optim_mat=False, optim_surf_range=None)

   Get optimizer parameter groups.

   :param lrs: Learning rates for [d, c, k, a]
   :type lrs: list
   :param decay: Decay for higher-order terms
   :type decay: float
   :param optim_mat: Optimize materials
   :type optim_mat: bool
   :param optim_surf_range: Surface indices to optimize
   :type optim_surf_range: list or None
   :return: Parameter list for optimizer
   :rtype: list

.. py:method:: GeoLens.get_optimizer(lrs=[1e-4, 1e-4, 1e-1, 1e-4], decay=0.01, optim_surf_range=None, optim_mat=False)

   Get Adam optimizer for lens parameters.

   :param lrs: Learning rates
   :type lrs: list
   :param decay: Decay factor
   :type decay: float
   :param optim_surf_range: Surfaces to optimize
   :type optim_surf_range: list or None
   :param optim_mat: Optimize materials
   :type optim_mat: bool
   :return: Adam optimizer
   :rtype: torch.optim.Adam

File I/O (GeoLensIO)
--------------------

JSON Format
~~~~~~~~~~~

.. py:method:: GeoLens.read_lens_json(filename="./test.json")

   Load lens from JSON file.

   :param filename: Path to .json file
   :type filename: str

.. py:method:: GeoLens.write_lens_json(filename="./test.json")

   Save lens to JSON file.

   :param filename: Output path
   :type filename: str

ZEMAX Format
~~~~~~~~~~~~

.. py:method:: GeoLens.read_lens_zmx(filename="./test.zmx")

   Load lens from ZEMAX .zmx file.

   :param filename: Path to .zmx file
   :type filename: str

.. py:method:: GeoLens.write_lens_zmx(filename="./test.zmx")

   Save lens to ZEMAX .zmx format.

   :param filename: Output path
   :type filename: str

Tolerance Analysis (GeoLensTolerance)
--------------------------------------

Setup
~~~~~

.. py:method:: GeoLens.init_tolerance(tolerance_params=None)

   Initialize tolerance parameters for manufacturing analysis.

   :param tolerance_params: Custom tolerance settings
   :type tolerance_params: dict or None

.. py:method:: GeoLens.sample_tolerance()

   Apply random manufacturing errors to all surfaces and refocus.

.. py:method:: GeoLens.zero_tolerance()

   Clear all manufacturing errors and restore nominal design.

Analysis Methods
~~~~~~~~~~~~~~~~

.. py:method:: GeoLens.tolerancing_sensitivity(tolerance_params=None)

   Compute tolerance sensitivity using first-order gradients.

   :param tolerance_params: Tolerance settings
   :type tolerance_params: dict or None
   :return: Sensitivity results dictionary
   :rtype: dict

.. py:method:: GeoLens.tolerancing_monte_carlo(trials=1000, tolerance_params=None)

   Perform Monte Carlo tolerance analysis.

   :param trials: Number of random trials
   :type trials: int
   :param tolerance_params: Tolerance settings
   :type tolerance_params: dict or None
   :return: Statistical tolerance results
   :rtype: dict

.. py:method:: GeoLens.tolerancing_wavefront(tolerance_params=None)

   Wavefront-based tolerance analysis.

   :param tolerance_params: Tolerance settings
   :type tolerance_params: dict or None
   :return: Wavefront tolerance results
   :rtype: dict

Visualization (GeoLensVis)
--------------------------

2D Layout
~~~~~~~~~

.. py:method:: GeoLens.draw_layout(filename, depth=float("inf"), zmx_format=True, multi_plot=False, lens_title=None, show=False)

   Draw 2D lens layout with ray paths.

   :param filename: Save filename (required)
   :type filename: str
   :param depth: Object depth
   :type depth: float
   :param zmx_format: Whether to use Zemax-style format
   :type zmx_format: bool
   :param multi_plot: Whether to create multiple plots (one per wavelength)
   :type multi_plot: bool
   :param lens_title: Plot title
   :type lens_title: str or None
   :param show: Display plot
   :type show: bool

.. py:method:: GeoLens.draw_lens_2d(ax, fig, plane="meridional", label=True)

   Draw lens surfaces in 2D.

   :param ax: Matplotlib axis
   :type ax: matplotlib.axes.Axes
   :param fig: Matplotlib figure
   :type fig: matplotlib.figure.Figure
   :param plane: "meridional" or "sagittal"
   :type plane: str
   :param label: Show surface labels
   :type label: bool

.. py:method:: GeoLens.draw_ray_2d(ray_o_record, ax, fig, color="b")

   Draw ray paths on 2D plot.

   :param ray_o_record: Ray intersection records
   :type ray_o_record: list
   :param ax: Matplotlib axis
   :type ax: matplotlib.axes.Axes
   :param fig: Matplotlib figure
   :type fig: matplotlib.figure.Figure
   :param color: Ray color
   :type color: str

2D Ray Sampling
~~~~~~~~~~~~~~~

.. py:method:: GeoLens.sample_parallel_2D(fov=0.0, num_rays=7, wvln=0.589, plane="meridional", entrance_pupil=True, depth=0.0)

   Sample 2D parallel rays for layout visualization.

   :param fov: Field angle in degrees
   :type fov: float
   :param num_rays: Number of rays
   :type num_rays: int
   :param wvln: Wavelength
   :type wvln: float
   :param plane: "meridional" or "sagittal"
   :type plane: str
   :param entrance_pupil: Use entrance pupil
   :type entrance_pupil: bool
   :param depth: Sampling depth
   :type depth: float
   :return: 2D ray object
   :rtype: Ray

.. py:method:: GeoLens.sample_point_source_2D(fov=0.0, num_rays=7, wvln=0.589, plane="meridional", depth=-10000.0)

   Sample 2D point source rays.

   :param fov: Field angle in degrees
   :type fov: float
   :param num_rays: Number of rays
   :type num_rays: int
   :param wvln: Wavelength
   :type wvln: float
   :param plane: "meridional" or "sagittal"
   :type plane: str
   :param depth: Object depth
   :type depth: float
   :return: 2D ray object
   :rtype: Ray

3D Visualization (GeoLensVis3D)
--------------------------------

.. py:method:: GeoLens.draw_lens_3d(filename=None, num_rays=5, view_angle=30, fov_list=None, depth=float("inf"), show=False)

   Draw 3D lens layout with rays using PyVista.

   :param filename: Save filename (.png)
   :type filename: str or None
   :param num_rays: Rays per field
   :type num_rays: int
   :param view_angle: Camera view angle
   :type view_angle: float
   :param fov_list: List of field angles to visualize
   :type fov_list: list or None
   :param depth: Object depth
   :type depth: float
   :param show: Display interactive window
   :type show: bool

.. py:method:: GeoLens.save_lens_obj(directory="./lens_mesh", num_rays=5, fov_list=None, depth=float("inf"))

   Save lens geometry and rays as .obj files.

   :param directory: Output directory
   :type directory: str
   :param num_rays: Rays per field
   :type num_rays: int
   :param fov_list: Field angles to export
   :type fov_list: list or None
   :param depth: Object depth
   :type depth: float

.. py:method:: GeoLens.create_mesh(num_rays=5, fov_list=None, depth=float("inf"))

   Create all lens/bridge/sensor/aperture meshes.

   :param num_rays: Rays per field
   :type num_rays: int
   :param fov_list: Field angles
   :type fov_list: list or None
   :param depth: Object depth
   :type depth: float
   :return: Dictionary with all mesh components
   :rtype: dict

.. py:method:: GeoLens.draw_layout_3d(filename=None, view_angle=30, show=False)

   Alternative 3D layout visualization.

   :param filename: Save filename
   :type filename: str or None
   :param view_angle: View angle
   :type view_angle: float
   :param show: Display plot
   :type show: bool

.. py:method:: GeoLens.create_barrier(r_extend=1.2, expand_factor=0.3, num_circle=64)

   Create 3D barrier mesh for lens system.

   :param r_extend: Radial extension factor
   :type r_extend: float
   :param expand_factor: Expansion factor
   :type expand_factor: float
   :param num_circle: Circle resolution
   :type num_circle: int
   :return: Barrier mesh vertices and faces
   :rtype: tuple

Static Methods
~~~~~~~~~~~~~~

.. py:method:: GeoLens.draw_mesh(plotter, mesh, color, opacity=1.0)
   :staticmethod:

   Draw a mesh to PyVista plotter.

   :param plotter: PyVista plotter
   :type plotter: pyvista.Plotter
   :param mesh: Mesh object
   :type mesh: CrossPoly
   :param color: RGB color [r, g, b]
   :type color: list
   :param opacity: Mesh opacity
   :type opacity: float

See Also
--------

- :doc:`lens` - Base Lens class documentation
- :doc:`optics` - Optical elements and surfaces
- :doc:`sensor` - Sensor and image processing
- :doc:`utils` - Utility functions

References
----------

1. Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
2. Jun Dai, Liqun Chen, Xinge Yang, Yuyao Hu, Jinwei Gu, Tianfan Xue, "Tolerance-Aware Deep Optics," arXiv:2502.04719, 2025.


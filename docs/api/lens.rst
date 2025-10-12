Lens API Reference
==================

This section documents the lens classes and their methods.

Base Lens Class
---------------

.. py:class:: Lens(device=None, dtype=torch.float32)

   Base class for all lens systems in DeepLens.

   :param device: Device to use ('cuda' or 'cpu')
   :param dtype: Data type for computations (default: torch.float32)

   .. py:method:: psf(depth=1000, spp=2048, method='wave', wavelength=None, field=[0, 0])

      Calculate the Point Spread Function.

      :param depth: Object distance in mm
      :param spp: Samples per pixel (number of rays)
      :param method: 'ray', 'wave', or 'coherent'
      :param wavelength: Wavelength in micrometers (None for RGB)
      :param field: Field position [x, y] normalized
      :return: PSF tensor [C, H, W]

   .. py:method:: render(img, depth=1000, spp=256, method='fft')

      Render an image through the lens system.

      :param img: Input image tensor [B, C, H, W]
      :param depth: Object distance in mm
      :param spp: Samples per pixel
      :param method: 'fft' or 'conv'
      :return: Rendered image tensor [B, C, H, W]

   .. py:method:: set_sensor(sensor_size=(10, 10), sensor_res=(512, 512))

      Set sensor dimensions.

      :param sensor_size: Physical sensor size (W, H) in mm
      :param sensor_res: Sensor resolution (W, H) in pixels

   .. py:method:: to(device)

      Move lens to specified device.

      :param device: 'cuda' or 'cpu'
      :return: self

   .. py:method:: parameters()

      Get optimizable parameters.

      :return: Iterator of torch.nn.Parameter

GeoLens
-------

.. py:class:: GeoLens(filename=None, sensor_res=(2000, 2000), sensor_size=(8.0, 8.0), device=None, dtype=torch.float32)

   Geometric lens system using ray tracing.

   :param filename: Path to lens file (.json, .zmx)
   :param sensor_res: Sensor resolution (W, H) in pixels
   :param sensor_size: Sensor size (W, H) in mm
   :param device: Device to use
   :param dtype: Data type

   **Attributes:**

   .. py:attribute:: surfaces

      List of optical surfaces

   .. py:attribute:: materials

      List of optical materials

   .. py:attribute:: foclen

      Focal length in mm (float)

   .. py:attribute:: fnum

      F-number (float)

   .. py:attribute:: enpd

      Entrance pupil diameter in mm (float)

   .. py:attribute:: hfov

      Horizontal field of view in degrees (float)

   **Methods:**

   .. py:method:: read_lens(filename)

      Load lens from file.

      :param filename: Path to .json or .zmx file

   .. py:method:: write_lens_json(filename)

      Save lens to JSON file.

      :param filename: Output file path

   .. py:method:: trace(ray)

      Trace rays through the lens.

      :param ray: Input Ray object
      :return: Output Ray object

   .. py:method:: sample_parallel_2D(R=5.0, M=256)

      Sample parallel ray bundle.

      :param R: Radius in mm
      :param M: Number of rays along one dimension
      :return: Ray object

   .. py:method:: sample_point_source(depth=1000, M=256, R=None)

      Sample rays from point source.

      :param depth: Source distance in mm
      :param M: Number of rays along one dimension
      :param R: Pupil radius (default: entrance_pupilr)
      :return: Ray object

   .. py:method:: sample_from_points(depth=1000, M=256, spp=100, field=[0, 0])

      Sample rays from off-axis point source.

      :param depth: Source distance in mm
      :param M: Grid resolution
      :param spp: Samples per point
      :param field: Field position [x, y]
      :return: Ray object

   .. py:method:: set_optimizer_params(params_dict)

      Configure which parameters to optimize.

      :param params_dict: Dictionary with keys 'radius', 'thickness', 'conic', 'ai', 'material'

      Example::

         lens.set_optimizer_params({
             'radius': True,
             'thickness': True,
             'ai': True
         })

   .. py:method:: calc_foclen()

      Calculate focal length.

      :return: Focal length in mm

   .. py:method:: calc_pupil()

      Calculate entrance pupil position and diameter.

   .. py:method:: calc_fov()

      Calculate field of view.

   .. py:method:: loss_constraint()

      Calculate constraint violations for optimization.

      :return: Constraint loss (scalar tensor)

   .. py:method:: plot_setup2D(M=10, plot_rays=True)

      Plot 2D cross-section of lens.

      :param M: Number of rays to plot
      :param plot_rays: Whether to show ray traces

   .. py:method:: plot_setup3D()

      Plot 3D visualization of lens.

   .. py:method:: plot_psf(psf, figsize=(10, 8))

      Visualize PSF.

      :param psf: PSF tensor [C, H, W]
      :param figsize: Figure size

   .. py:method:: plot_psf_map(psf_map, figsize=(15, 10))

      Visualize PSF across field.

      :param psf_map: PSF map tensor
      :param figsize: Figure size

   .. py:method:: analysis_rms_spot()

      Analyze RMS spot size across field.

      :return: RMS spot size map

   .. py:method:: analysis_distortion()

      Analyze geometric distortion.

      :return: Distortion map

   .. py:method:: analysis_mtf(frequency=50)

      Analyze Modulation Transfer Function.

      :param frequency: Spatial frequency in lp/mm
      :return: MTF map

PSFNetLens
----------

.. py:class:: PSFNetLens(ckpt_path, device='cuda')

   Neural surrogate lens model.

   :param ckpt_path: Path to checkpoint file
   :param device: Device to use

   **Attributes:**

   .. py:attribute:: foclen

      Focal length in mm (float)

   .. py:attribute:: fnum

      F-number (float)

   .. py:attribute:: pixel_size

      Pixel size in mm (float)

   **Methods:**

   .. py:method:: psf(depth, field=[0, 0], wvln=0.550)

      Fast PSF prediction.

      :param depth: Object distance in mm
      :param field: Field position [x, y]
      :param wvln: Wavelength in micrometers
      :return: PSF tensor [1, H, W]

   .. py:method:: render(img, depth=1000)

      Fast image rendering.

      :param img: Input image [B, C, H, W]
      :param depth: Object distance in mm
      :return: Rendered image [B, C, H, W]

HybridLens
----------

.. py:class:: HybridLens(filename=None, sensor_res=(2000, 2000), sensor_size=(8.0, 8.0), wave_method='asm', device=None)

   Hybrid refractive-diffractive lens system.

   :param filename: Path to lens file
   :param sensor_res: Sensor resolution
   :param sensor_size: Sensor size in mm
   :param wave_method: Wave propagation method ('asm' or 'fresnel')
   :param device: Device to use

   **Methods:**

   .. py:method:: trace_hybrid(ray)

      Trace through hybrid system (ray + wave).

      :param ray: Input Ray object
      :return: Complex field at sensor

   .. py:method:: render(img, depth=1000, spp=512)

      Render image through hybrid lens.

      :param img: Input image
      :param depth: Object distance
      :param spp: Samples per pixel
      :return: Rendered image

ParaxialLens
------------

.. py:class:: ParaxialLens(foclen=50.0, fnum=2.0, sensor_res=(512, 512), device='cuda')

   Paraxial (thin lens) model.

   :param foclen: Focal length in mm
   :param fnum: F-number
   :param sensor_res: Sensor resolution
   :param device: Device to use

   **Methods:**

   .. py:method:: render(img, depth, focus_depth=None)

      Render with defocus blur.

      :param img: Input image
      :param depth: Object distance in mm
      :param focus_depth: Focus distance (default: depth)
      :return: Blurred image

Camera
------

.. py:class:: Camera(lens, sensor=None, isp=None, device='cuda')

   Complete camera system.

   :param lens: Lens object
   :param sensor: Sensor object
   :param isp: ISP pipeline object
   :param device: Device to use

   **Attributes:**

   .. py:attribute:: lens

      Lens system

   .. py:attribute:: sensor

      Image sensor

   .. py:attribute:: isp

      Image signal processor

   **Methods:**

   .. py:method:: capture(scene, depth, exposure_time=0.01, iso=100, auto_focus=False)

      Capture image end-to-end.

      :param scene: Scene image tensor
      :param depth: Object distance or depth map
      :param exposure_time: Exposure time in seconds
      :param iso: ISO sensitivity
      :param auto_focus: Enable auto-focus
      :return: Processed image

   .. py:method:: capture_raw(scene, depth, exposure_time=0.01, iso=100)

      Capture raw sensor data.

      :param scene: Scene image
      :param depth: Object distance
      :param exposure_time: Exposure time
      :param iso: ISO setting
      :return: Raw sensor data

   .. py:method:: set_focus(focus_depth)

      Set focus distance.

      :param focus_depth: Focus distance in mm

   .. py:method:: auto_focus(scene, depth)

      Automatic focus adjustment.

      :param scene: Scene for focus detection
      :param depth: Depth information

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    from deeplens import GeoLens
    
    # Create lens
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Calculate PSF
    psf = lens.psf(depth=1000, spp=2048)
    
    # Render image
    img_rendered = lens.render(img, depth=1000)

Lens Optimization
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.optim as optim
    from deeplens.optics import SpotLoss
    
    # Setup optimization
    lens.set_optimizer_params({'radius': True, 'thickness': True})
    optimizer = optim.Adam(lens.parameters(), lr=0.01)
    loss_fn = SpotLoss()
    
    # Optimization loop
    for i in range(1000):
        optimizer.zero_grad()
        ray = lens.sample_point_source(depth=1e4, M=256)
        ray_out = lens.trace(ray)
        loss = loss_fn(ray_out) + lens.loss_constraint()
        loss.backward()
        optimizer.step()

Fast PSF Prediction
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import PSFNetLens
    
    # Load pre-trained model
    lens = PSFNetLens(
        ckpt_path='./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth'
    )
    
    # Fast PSF calculation
    psf = lens.psf(depth=1000, field=[0, 0.5])
    
    # 100x faster than ray tracing!

Camera System
^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import Camera, GeoLens
    from deeplens.sensor import RGBSensor, ISP
    
    # Create camera
    camera = Camera(
        lens=GeoLens(filename='lens.json'),
        sensor=RGBSensor(resolution=(1920, 1080)),
        isp=ISP()
    )
    
    # Capture image
    img = camera.capture(scene, depth=1000, exposure_time=0.01)

See Also
--------

* :doc:`../user_guide/lens_systems` - Detailed lens system guide
* :doc:`../tutorials` - Step-by-step tutorials
* :doc:`../examples/automated_lens_design` - Optimization examples


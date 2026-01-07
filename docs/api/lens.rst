Lens API Reference
==================

This section documents the lens classes and their methods.

Base Lens Class
---------------

.. py:class:: Lens(device=None, dtype=torch.float32)

   Base class for all lens systems in DeepLens.

   :param device: Device to use ('cuda' or 'cpu')
   :param dtype: Data type for computations (default: torch.float32)

   .. py:method:: psf(points, wvln=0.589, ks=64, **kwargs)

      Compute monochrome point PSF. This function should be differentiable.

      :param points: Point positions, shape [N, 3] or [3], normalized xy in [-1, 1], z < 0
      :type points: torch.Tensor
      :param wvln: Wavelength in micrometers
      :type wvln: float
      :param ks: Kernel size
      :type ks: int
      :return: PSF tensor [ks, ks] or [N, ks, ks]
      :rtype: torch.Tensor

   .. py:method:: render(img_obj, depth=-20000.0, method='psf_patch', **kwargs)

      Differentiable image simulation through the lens.

      :param img_obj: Input image tensor [B, C, H, W]
      :type img_obj: torch.Tensor
      :param depth: Object depth in mm
      :type depth: float
      :param method: Rendering method - 'psf_map' or 'psf_patch'
      :type method: str
      :param kwargs: Additional method-specific arguments (psf_grid, psf_ks, psf_center)
      :return: Rendered image tensor [B, C, H, W]
      :rtype: torch.Tensor

   .. py:method:: set_sensor(sensor_size, sensor_res)

      Set sensor dimensions.

      :param sensor_size: Physical sensor size (W, H) in mm
      :type sensor_size: tuple
      :param sensor_res: Sensor resolution (W, H) in pixels
      :type sensor_res: tuple

   .. py:method:: to(device)

      Move lens to specified device.

      :param device: 'cuda' or 'cpu'
      :return: self

   .. py:method:: parameters()

      Get optimizable parameters.

      :return: Iterator of torch.nn.Parameter

GeoLens
-------

.. py:class:: GeoLens(filename=None, device=None, dtype=torch.float32)

   Geometric lens system using ray tracing.

   :param filename: Path to lens file (.json, .zmx)
   :param device: Device to use
   :param dtype: Data type

   .. note::
      Sensor size and resolution are read from the lens file. If not provided,
      defaults of 8mm x 8mm and 2000x2000 pixels will be used.

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

   .. py:method:: sample_parallel_2D(fov=0.0, num_rays=7, wvln=0.589, plane='meridional', entrance_pupil=True, depth=0.0)

      Sample 2D parallel rays for layout visualization.

      :param fov: Field angle in degrees
      :type fov: float
      :param num_rays: Number of rays
      :type num_rays: int
      :param wvln: Wavelength in micrometers
      :type wvln: float
      :param plane: 'meridional' or 'sagittal'
      :type plane: str
      :param entrance_pupil: Use entrance pupil
      :type entrance_pupil: bool
      :param depth: Sampling depth
      :type depth: float
      :return: Ray object with shape [num_rays, 3]
      :rtype: Ray

   .. py:method:: sample_point_source(fov_x=[0.0], fov_y=[0.0], depth=-20000.0, num_rays=16384, wvln=0.589, entrance_pupil=True, scale_pupil=1.0)

      Sample point source rays from object space with given field angles.

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

   .. py:method:: sample_from_points(points=[[0.0, 0.0, -10000.0]], num_rays=16384, wvln=0.589, scale_pupil=1.0)

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

.. py:class:: PSFNetLens(lens_path, in_chan=3, psf_chan=3, model_name='mlp_conv', kernel_size=64)

   Neural surrogate lens model that represents the PSF using a neural network.

   :param lens_path: Path to the lens JSON file
   :type lens_path: str
   :param in_chan: Number of input channels (fov, depth, foc_dist)
   :type in_chan: int
   :param psf_chan: Number of output PSF channels (RGB)
   :type psf_chan: int
   :param model_name: Network architecture ('mlp' or 'mlpconv')
   :type model_name: str
   :param kernel_size: PSF kernel size
   :type kernel_size: int

   .. note::
      Sensor size and resolution are read from the lens file. If not provided,
      GeoLens defaults will be used.

   **Attributes:**

   .. py:attribute:: lens

      Embedded GeoLens object

   .. py:attribute:: psfnet

      Neural network for PSF prediction

   .. py:attribute:: pixel_size

      Pixel size in mm (float)

   **Methods:**

   .. py:method:: psf_rgb(points, ks=64)

      Fast RGB PSF prediction using neural network.

      :param points: Point positions [N, 3], normalized xy in [-1, 1], z is depth in mm
      :type points: torch.Tensor
      :param ks: Kernel size
      :type ks: int
      :return: PSF tensor [N, 3, ks, ks]
      :rtype: torch.Tensor

   .. py:method:: render_rgbd(img, depth, foc_dist, ks=64, high_res=False)

      Render image with depth map using per-pixel PSF convolution.

      :param img: Input image [1, C, H, W]
      :type img: torch.Tensor
      :param depth: Depth map [1, H, W] in mm
      :type depth: torch.Tensor
      :param foc_dist: Focus distance in mm
      :type foc_dist: float
      :param ks: PSF kernel size
      :type ks: int
      :param high_res: Use high resolution rendering
      :type high_res: bool
      :return: Rendered image [1, C, H, W]
      :rtype: torch.Tensor

   .. py:method:: refocus(foc_dist)

      Refocus the lens to a given focus distance.

      :param foc_dist: Focus distance in mm
      :type foc_dist: float

   .. py:method:: load_net(net_path)

      Load pretrained network weights.

      :param net_path: Path to network checkpoint
      :type net_path: str

   .. py:method:: train_psfnet(iters=100000, bs=128, lr=5e-5, evaluate_every=500, spp=16384, concentration_factor=2.0, result_dir='./results/psfnet')

      Train the PSF surrogate network.

      :param iters: Number of training iterations
      :type iters: int
      :param bs: Batch size
      :type bs: int
      :param lr: Learning rate
      :type lr: float
      :param evaluate_every: Evaluation interval
      :type evaluate_every: int
      :param spp: Samples per pixel for ray tracing
      :type spp: int
      :param concentration_factor: Concentration factor for training data sampling
      :type concentration_factor: float
      :param result_dir: Directory to save results
      :type result_dir: str

HybridLens
----------

.. py:class:: HybridLens(filename=None, device=None, dtype=torch.float64)

   Hybrid refractive-diffractive lens using a differentiable ray-wave model.

   :param filename: Path to hybrid-lens JSON file. If None, create empty hybrid lens
   :type filename: str or None
   :param device: Computing device ('cuda' or 'cpu'). If None, auto-selects
   :type device: str or None
   :param dtype: Data type for computations (default: torch.float64)
   :type dtype: torch.dtype

   .. note::
      Sensor size and resolution are read from the lens file (GeoLens section).
      If not provided, defaults of 8mm x 8mm and 2000x2000 pixels will be used.

   **Methods:**

   .. py:method:: psf(points=[0.0, 0.0, -10000.0], ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT)

      Single point monochromatic PSF using ray-wave model.

      :param points: Point source position [x, y, z] (normalized x,y in [-1, 1], z < 0)
      :type points: list or torch.Tensor
      :param ks: Output PSF kernel size
      :type ks: int or None
      :param wvln: Wavelength in micrometers
      :type wvln: float
      :param spp: Rays per point for coherent tracing (>= 1e6 recommended)
      :type spp: int
      :return: Normalized PSF patch [ks, ks]
      :rtype: torch.Tensor

   .. py:method:: draw_layout(save_name='./DOELens.png', depth=-10000.0, ax=None, fig=None)

      Draw the hybrid system layout with ray-tracing and wave-propagation.

      :param save_name: Output figure path
      :type save_name: str
      :param depth: Object depth for ray bundles (mm)
      :type depth: float
      :param ax: Optional matplotlib axis
      :type ax: matplotlib.axes.Axes or None
      :param fig: Optional matplotlib figure
      :type fig: matplotlib.figure.Figure or None

ParaxialLens
------------

.. py:class:: ParaxialLens(foclen, fnum, sensor_size=None, sensor_res=None, device='cpu')

   Paraxial (thin lens) model.

   :param foclen: Focal length in mm
   :param fnum: F-number
   :param sensor_size: Sensor size (W, H) in mm. If None, defaults to (8.0, 8.0)
   :param sensor_res: Sensor resolution (W, H) in pixels. If None, defaults to (2000, 2000)
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


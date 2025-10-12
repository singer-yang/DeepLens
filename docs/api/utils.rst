Utilities API Reference
=======================

Utility functions and helper tools.

Image Processing Utilities
---------------------------

batch_psnr
^^^^^^^^^^

.. py:function:: deeplens.utils.batch_psnr(img1, img2, max_val=1.0)

   Calculate PSNR between image batches.

   :param img1: First image batch [B, C, H, W]
   :param img2: Second image batch [B, C, H, W]
   :param max_val: Maximum pixel value
   :return: PSNR value in dB

batch_ssim
^^^^^^^^^^

.. py:function:: deeplens.utils.batch_ssim(img1, img2, window_size=11)

   Calculate SSIM between image batches.

   :param img1: First image batch [B, C, H, W]
   :param img2: Second image batch [B, C, H, W]
   :param window_size: SSIM window size
   :return: SSIM value [0, 1]

img2batch
^^^^^^^^^

.. py:function:: deeplens.utils.img2batch(img, batch_size=1)

   Convert image to batch format.

   :param img: Image [C, H, W]
   :param batch_size: Batch size
   :return: Batched image [B, C, H, W]

Visualization Utilities
-----------------------

plot_image
^^^^^^^^^^

.. py:function:: deeplens.utils.plot_image(img, title='', figsize=(10, 8), colorbar=True)

   Plot image with matplotlib.

   :param img: Image tensor [C, H, W] or [H, W]
   :param title: Plot title
   :param figsize: Figure size
   :param colorbar: Show colorbar

plot_comparison
^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.plot_comparison(images, titles=None, figsize=(15, 5))

   Plot multiple images side by side.

   :param images: List of images
   :param titles: List of titles
   :param figsize: Figure size

save_image_grid
^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.save_image_grid(images, filename, nrow=4, padding=2)

   Save images as grid.

   :param images: Image tensor [B, C, H, W]
   :param filename: Output filename
   :param nrow: Number of images per row
   :param padding: Padding between images

Tensor Utilities
----------------

tensor_to_numpy
^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.tensor_to_numpy(tensor)

   Convert PyTorch tensor to numpy array.

   :param tensor: Input tensor
   :return: Numpy array

numpy_to_tensor
^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.numpy_to_tensor(array, device='cuda')

   Convert numpy array to PyTorch tensor.

   :param array: Numpy array
   :param device: Target device
   :return: PyTorch tensor

normalize
^^^^^^^^^

.. py:function:: deeplens.utils.normalize(tensor, min_val=None, max_val=None)

   Normalize tensor to [0, 1].

   :param tensor: Input tensor
   :param min_val: Minimum value (None for auto)
   :param max_val: Maximum value (None for auto)
   :return: Normalized tensor

File I/O Utilities
------------------

load_image
^^^^^^^^^^

.. py:function:: deeplens.utils.load_image(filename, resize=None, device='cuda')

   Load image from file.

   :param filename: Image file path
   :param resize: Target size (W, H) or None
   :param device: Target device
   :return: Image tensor [1, C, H, W]

save_image
^^^^^^^^^^

.. py:function:: deeplens.utils.save_image(tensor, filename, normalize=True)

   Save tensor as image.

   :param tensor: Image tensor
   :param filename: Output filename
   :param normalize: Normalize to [0, 1]

load_depth
^^^^^^^^^^

.. py:function:: deeplens.utils.load_depth(filename, scale=1.0, device='cuda')

   Load depth map from file.

   :param filename: Depth file path (EXR, PNG, etc.)
   :param scale: Depth scaling factor
   :param device: Target device
   :return: Depth tensor [1, 1, H, W]

save_depth
^^^^^^^^^^

.. py:function:: deeplens.utils.save_depth(depth, filename, scale=1.0)

   Save depth map to file.

   :param depth: Depth tensor
   :param filename: Output filename
   :param scale: Depth scaling factor

Configuration Utilities
-----------------------

load_config
^^^^^^^^^^^

.. py:function:: deeplens.utils.load_config(filename)

   Load YAML configuration file.

   :param filename: Config file path
   :return: Dictionary with config

save_config
^^^^^^^^^^^

.. py:function:: deeplens.utils.save_config(config, filename)

   Save configuration to YAML.

   :param config: Configuration dictionary
   :param filename: Output file path

Logging Utilities
-----------------

setup_logger
^^^^^^^^^^^^

.. py:function:: deeplens.utils.setup_logger(name, log_file=None, level='INFO')

   Setup logger.

   :param name: Logger name
   :param log_file: Log file path (None for console only)
   :param level: Logging level
   :return: Logger object

log_metrics
^^^^^^^^^^^

.. py:function:: deeplens.utils.log_metrics(metrics, epoch, logger=None)

   Log training metrics.

   :param metrics: Dictionary of metrics
   :param epoch: Current epoch
   :param logger: Logger object (None for print)

TensorBoard Utilities
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.utils.TensorBoardLogger(log_dir='runs')

   TensorBoard logger wrapper.

   :param log_dir: Directory for logs

   .. py:method:: add_scalar(tag, value, step)

      Log scalar value.

      :param tag: Metric name
      :param value: Scalar value
      :param step: Step number

   .. py:method:: add_image(tag, image, step)

      Log image.

      :param tag: Image name
      :param image: Image tensor
      :param step: Step number

   .. py:method:: add_images(tag, images, step)

      Log multiple images.

      :param tag: Images name
      :param images: Image batch
      :param step: Step number

Device Management
-----------------

get_device
^^^^^^^^^^

.. py:function:: deeplens.utils.get_device(device=None)

   Get PyTorch device.

   :param device: 'cuda', 'cpu', or None (auto-detect)
   :return: torch.device

get_gpu_memory
^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.get_gpu_memory()

   Get GPU memory usage.

   :return: Dictionary with allocated and cached memory

clear_cache
^^^^^^^^^^^

.. py:function:: deeplens.utils.clear_cache()

   Clear PyTorch cache.

set_seed
^^^^^^^^

.. py:function:: deeplens.utils.set_seed(seed=42)

   Set random seed for reproducibility.

   :param seed: Random seed

Math Utilities
--------------

deg2rad
^^^^^^^

.. py:function:: deeplens.utils.deg2rad(degrees)

   Convert degrees to radians.

   :param degrees: Angle in degrees
   :return: Angle in radians

rad2deg
^^^^^^^

.. py:function:: deeplens.utils.rad2deg(radians)

   Convert radians to degrees.

   :param radians: Angle in radians
   :return: Angle in degrees

safe_divide
^^^^^^^^^^^

.. py:function:: deeplens.utils.safe_divide(a, b, eps=1e-8)

   Safe division avoiding divide-by-zero.

   :param a: Numerator
   :param b: Denominator
   :param eps: Small epsilon value
   :return: a / (b + eps)

Performance Utilities
---------------------

timer
^^^^^

.. py:class:: deeplens.utils.Timer()

   Context manager for timing code.

   **Example:**

   .. code-block:: python

      with Timer() as t:
          # Code to time
          result = expensive_function()
      print(f"Elapsed: {t.elapsed:.3f} seconds")

ProgressBar
^^^^^^^^^^^

.. py:class:: deeplens.utils.ProgressBar(total, desc='')

   Progress bar for loops.

   :param total: Total iterations
   :param desc: Description

   .. py:method:: update(n=1)

      Update progress.

      :param n: Increment

   **Example:**

   .. code-block:: python

      pbar = ProgressBar(total=100, desc='Processing')
      for i in range(100):
          # Do work
          pbar.update()

profile_memory
^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.profile_memory(func)

   Decorator to profile memory usage.

   :param func: Function to profile
   :return: Wrapped function

   **Example:**

   .. code-block:: python

      @profile_memory
      def my_function():
          # Function code
          pass

Constants
---------

.. py:data:: DEFAULT_WAVE

   Default wavelength (550nm, green)

.. py:data:: WAVE_RGB

   RGB wavelengths [486, 550, 656] nm

.. py:data:: SPP_RENDER

   Default samples per pixel for rendering (256)

.. py:data:: SPP_PSF

   Default samples per pixel for PSF (2048)

.. py:data:: DEPTH

   Default object depth (1000mm = 1m)

.. py:data:: EPSILON

   Small epsilon value (1e-8)

Examples
--------

Image Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import batch_psnr, batch_ssim
    import torch
    
    # Compare images
    img1 = torch.rand(4, 3, 256, 256)
    img2 = torch.rand(4, 3, 256, 256)
    
    psnr = batch_psnr(img1, img2)
    ssim = batch_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

Visualization
^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import plot_comparison
    
    images = [img1, img2, img3]
    titles = ['Original', 'Degraded', 'Restored']
    plot_comparison(images, titles)

Configuration
^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import load_config, save_config
    
    # Load config
    config = load_config('config.yml')
    
    # Modify and save
    config['learning_rate'] = 0.001
    save_config(config, 'config_modified.yml')

Logging
^^^^^^^

.. code-block:: python

    from deeplens.utils import setup_logger
    
    logger = setup_logger('training', 'train.log')
    logger.info('Training started')
    logger.debug('Debug information')
    logger.warning('Warning message')

TensorBoard
^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import TensorBoardLogger
    
    tb_logger = TensorBoardLogger('runs/experiment1')
    
    for epoch in range(100):
        # Training
        loss = train_one_epoch()
        
        # Log metrics
        tb_logger.add_scalar('loss/train', loss, epoch)
        
        # Log images
        if epoch % 10 == 0:
            tb_logger.add_image('output', output_img, epoch)

Timing
^^^^^^

.. code-block:: python

    from deeplens.utils import Timer
    
    with Timer() as t:
        psf = lens.psf(depth=1000, spp=2048)
    
    print(f"PSF calculation took {t.elapsed:.3f} seconds")

Progress Bar
^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import ProgressBar
    
    pbar = ProgressBar(total=len(dataloader), desc='Training')
    
    for batch in dataloader:
        # Process batch
        loss = train_step(batch)
        pbar.update()

Reproducibility
^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import set_seed
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Now all random operations are deterministic
    model = create_model()
    # Training will be reproducible

See Also
--------

* :doc:`../tutorials` - Usage examples in context
* :doc:`lens` - Lens system API
* :doc:`../user_guide/lens_systems` - User guides


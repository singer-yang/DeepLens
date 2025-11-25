Utilities API Reference
=======================

Utility functions and helper tools.

Image Processing Utilities
---------------------------

batch_psnr
^^^^^^^^^^

.. py:function:: deeplens.utils.batch_psnr(pred, target, max_val=1.0, eps=1e-8)

   Calculate PSNR between image batches.

   :param pred: Predicted image batch [B, C, H, W]
   :param target: Target image batch [B, C, H, W]
   :param max_val: Maximum pixel value
   :param eps: Small constant for numerical stability
   :return: PSNR value in dB

batch_ssim
^^^^^^^^^^

.. py:function:: deeplens.utils.batch_ssim(img, img_clean)

   Calculate SSIM between image batches.

   :param img: Input image batch [B, C, H, W]
   :param img_clean: Reference image batch [B, C, H, W]
   :return: SSIM value [0, 1]

batch_LPIPS
^^^^^^^^^^^

.. py:function:: deeplens.utils.batch_LPIPS(img, img_clean)

   Compute LPIPS loss for image batch.

   :param img: Input image batch
   :param img_clean: Reference image batch
   :return: LPIPS distance

img2batch
^^^^^^^^^

.. py:function:: deeplens.utils.img2batch(img)

   Convert image to batch format.

   :param img: Image tensor (H, W, C) or (C, H, W), or numpy array
   :return: Batched image [1, C, H, W]

Image Normalization
^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.normalize_ImageNet(batch)

   Normalize dataset by ImageNet statistics.

   :param batch: Input image batch
   :return: Normalized batch

.. py:function:: deeplens.utils.denormalize_ImageNet(batch)

   Convert normalized images back to original range.

   :param batch: Normalized batch
   :return: Denormalized batch

Interpolation
-------------

interp1d
^^^^^^^^

.. py:function:: deeplens.utils.interp1d(query, key, value, mode="linear")

   Interpolate 1D query points to the key points.

   :param query: Query points [N, 1]
   :param key: Key points [M, 1]
   :param value: Value at key points [M, ...]
   :param mode: Interpolation mode
   :return: Interpolated value [N, ...]

grid_sample_xy
^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.grid_sample_xy(input, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False)

   Grid sample using xy-coordinate grid [-1, 1].

   :param input: Input tensor [B, C, H, W]
   :param grid_xy: Grid xy coordinates [B, H, W, 2]
   :return: Sampled tensor [B, C, H, W]

Video Utilities
---------------

create_video_from_images
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: deeplens.utils.create_video_from_images(image_folder, output_video_path, fps=30)

   Create a video from a folder of images.

   :param image_folder: Path to folder containing images
   :param output_video_path: Output video path
   :param fps: Frames per second

Logging and Setup
-----------------

set_logger
^^^^^^^^^^

.. py:function:: deeplens.utils.set_logger(dir="./")

   Setup logger.

   :param dir: Log directory

gpu_init
^^^^^^^^

.. py:function:: deeplens.utils.gpu_init(gpu=0)

   Initialize device and data type.

   :param gpu: GPU index
   :return: torch.device

set_seed
^^^^^^^^

.. py:function:: deeplens.utils.set_seed(seed=0)

   Set random seed for reproducibility.

   :param seed: Random seed

Constants
---------

.. py:data:: DEFAULT_WAVE

   Default wavelength (0.58756180 um)

.. py:data:: WAVE_RGB

   RGB wavelengths [0.65627250, 0.58756180, 0.48613270] um

.. py:data:: SPP_PSF

   Default samples per pixel for PSF (16384)

.. py:data:: SPP_COHERENT

   Samples per pixel for coherent calculation (~16.7M = 2^24)

.. py:data:: SPP_CALC

   Samples for computation (1024)

.. py:data:: SPP_RENDER

   Samples per pixel for rendering (32)

.. py:data:: DEPTH

   Default object depth (-20000.0 mm)

.. py:data:: PSF_KS

   Default kernel size for PSF calculation (64)

.. py:data:: EPSILON

   Small constant to avoid division by zero (1e-9)


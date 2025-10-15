Sensor API Reference
====================

This section documents sensor simulation and image signal processing.

Sensor Classes
--------------

RGBSensor
^^^^^^^^^

.. py:class:: deeplens.sensor.RGBSensor(sensor_file)

   RGB sensor with Bayer color filter array driven by a configuration file.

   :param sensor_file: Path to a JSON file describing sensor parameters

   The constructor reads key properties from ``sensor_file`` (bit depth, black level,
   Bayer pattern, noise statistics, optional white balance/color matrix/gamma, and
   optional spectral responses). Internally it builds an :class:`deeplens.sensor.isp.InvertibleISP`.

   **Key attributes:**

   - ``size``: Physical sensor size (W, H) in mm
   - ``res``: Sensor resolution (W, H) in pixels
   - ``bit``: ADC bit depth
   - ``black_level``: Black level offset in DN
   - ``bayer_pattern``: Bayer pattern string (e.g., ``"rggb"``)
   - ``iso_base``, ``read_noise_std``, ``shot_noise_std_alpha``, ``shot_noise_std_beta``
   - ``isp``: :class:`deeplens.sensor.isp.InvertibleISP`

   **Selected methods:**

   .. py:method:: forward(img_nbit, iso)

      Simulate sensor noise and run ISP.

      :param img_nbit: RAW RGB or Bayer-space tensor [B, 3, H, W] in n-bit DN
      :param iso: ISO value(s) [B]
      :return: RGB image [B, 3, H, W] in [0, 1]

   .. py:method:: unprocess(image, in_type='rgb')

      Convert RGB to unbalanced RAW RGB (inverse gamma/CCM/AWB).

   .. py:method:: linrgb2bayer(img_linrgb)

      Convert linear RGB [0, 1] to n-bit Bayer [~black_level, 2**bit - 1].

   .. py:method:: process2rgb(image, in_type='rggb')

      Process Bayer or RGGB to RGB using the internal ISP.

   .. py:method:: bayer2rggb(bayer_nbit) / rggb2bayer(rggb)

      Pack/unpack between Bayer [B,1,H,W] and RGGB [B,4,H/2,W/2].

MonoSensor
^^^^^^^^^^

.. py:class:: deeplens.sensor.MonoSensor(resolution=(2048, 2048), pixel_size=3.45e-3, bit_depth=14, qe=0.7, device='cuda')

   Monochrome sensor without color filter.

   :param resolution: Sensor resolution
   :param pixel_size: Pixel size in mm
   :param bit_depth: ADC bit depth
   :param qe: Quantum efficiency
   :param device: Device

   **Methods:**

   .. py:method:: capture(irradiance, exposure_time=0.01, iso=100)

      Capture monochrome image.

      :param irradiance: Irradiance [W/m²] [H, W]
      :param exposure_time: Exposure time
      :param iso: ISO
      :return: Sensor data [H, W]

EventSensor
^^^^^^^^^^^

.. py:class:: deeplens.sensor.EventSensor(resolution=(640, 480), pixel_size=18.5e-3, threshold=0.1, refractory_period=1e-3, device='cuda')

   Event-based (DVS) sensor.

   :param resolution: Sensor resolution
   :param pixel_size: Pixel size in mm
   :param threshold: Contrast threshold for events
   :param refractory_period: Refractory period in seconds
   :param device: Device

   **Methods:**

   .. py:method:: capture_events(frame_sequence, timestamps)

      Generate events from frame sequence.

      :param frame_sequence: Sequence of frames [T, H, W]
      :param timestamps: Time for each frame [T]
      :return: Events (x, y, t, p) where p is polarity

ISP Pipeline
------------

ISP
^^^

.. py:class:: deeplens.sensor.ISP(demosaic_method='bilinear', white_balance=True, color_correction=True, gamma_correction=True, denoise=False, sharpen=False, device='cuda')

   Complete Image Signal Processing pipeline.

   :param demosaic_method: Demosaicing algorithm
   :param white_balance: Enable white balance
   :param color_correction: Enable color correction
   :param gamma_correction: Enable gamma correction
   :param denoise: Enable denoising
   :param sharpen: Enable sharpening
   :param device: Device

   **Methods:**

   .. py:method:: forward(raw_image)

      Process raw sensor data.

      :param raw_image: Raw Bayer image [H, W]
      :return: Processed RGB image [3, H, W]

   .. py:method:: set_ccm(matrix)

      Set color correction matrix.

      :param matrix: 3x3 color correction matrix

   .. py:method:: set_wb_gains(gains)

      Set white balance gains.

      :param gains: [R, G, B] gains

ISP Modules
-----------

BlackLevel
^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.BlackLevel(level=64, device='cuda')

   Black level correction.

   :param level: Black level offset
   :param device: Device

   .. py:method:: forward(raw)

      Apply black level correction.

      :param raw: Raw sensor data
      :return: Corrected data

LensShadingCorrection
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.LensShadingCorrection(resolution, center=None, falloff=0.3, device='cuda')

   Lens shading correction for vignetting.

   :param resolution: Image resolution (W, H)
   :param center: Optical center [cx, cy] (default: image center)
   :param falloff: Vignetting falloff factor
   :param device: Device

   .. py:method:: forward(raw)

      Apply lens shading correction.

      :param raw: Raw image
      :return: Corrected image

DeadPixelCorrection
^^^^^^^^^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.DeadPixelCorrection(threshold=0.1, method='median', device='cuda')

   Dead and hot pixel correction.

   :param threshold: Detection threshold
   :param method: Correction method ('median', 'mean')
   :param device: Device

   .. py:method:: forward(raw)

      Correct dead pixels.

      :param raw: Raw image
      :return: Corrected image

WhiteBalance
^^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.WhiteBalance(method='gray_world', gains=None, device='cuda')

   White balance correction.

   :param method: 'gray_world', 'white_patch', or 'manual'
   :param gains: Manual gains [R, G, B] (for manual mode)
   :param device: Device

   .. py:method:: forward(raw)

      Apply white balance.

      :param raw: Raw Bayer image
      :return: Balanced image

   .. py:method:: estimate_gains(raw)

      Estimate white balance gains.

      :param raw: Raw image
      :return: Estimated [R, G, B] gains

Demosaic
^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.Demosaic(method='bilinear', device='cuda')

   Bayer demosaicing.

   :param method: 'bilinear', 'malvar', 'menon', or 'ahd'
   :param device: Device

   .. py:method:: forward(bayer)

      Demosaic Bayer pattern to RGB.

      :param bayer: Bayer image [H, W]
      :return: RGB image [3, H, W]

**Available Methods:**

* **bilinear**: Fast bilinear interpolation
* **malvar**: Edge-aware interpolation
* **menon**: High-quality edge-directed
* **ahd**: Adaptive homogeneity-directed

ColorMatrix
^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.ColorMatrix(matrix=None, device='cuda')

   Color correction matrix.

   :param matrix: 3x3 correction matrix (default: identity)
   :param device: Device

   .. py:method:: forward(rgb)

      Apply color correction.

      :param rgb: RGB image [3, H, W]
      :return: Corrected RGB [3, H, W]

GammaCorrection
^^^^^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.GammaCorrection(gamma=2.2, method='power', device='cuda')

   Gamma correction for display.

   :param gamma: Gamma value
   :param method: 'power', 'srgb', or 'log'
   :param device: Device

   .. py:method:: forward(linear_rgb)

      Apply gamma correction.

      :param linear_rgb: Linear RGB [3, H, W]
      :return: Gamma-corrected RGB [3, H, W]

Denoise
^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.Denoise(method='bilateral', strength=0.5, device='cuda')

   Image denoising.

   :param method: 'bilateral', 'nlm', or 'bm3d'
   :param strength: Denoising strength
   :param device: Device

   .. py:method:: forward(rgb)

      Denoise image.

      :param rgb: RGB image [3, H, W]
      :return: Denoised image [3, H, W]

ColorSpace
^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.ColorSpace(device='cuda')

   Color space conversions.

   :param device: Device

   .. py:method:: rgb_to_yuv(rgb)

      RGB to YUV conversion.

      :param rgb: RGB image [3, H, W]
      :return: YUV image [3, H, W]

   .. py:method:: yuv_to_rgb(yuv)

      YUV to RGB conversion.

      :param yuv: YUV image [3, H, W]
      :return: RGB image [3, H, W]

   .. py:method:: rgb_to_hsv(rgb)

      RGB to HSV conversion.

      :param rgb: RGB image [3, H, W]
      :return: HSV image [3, H, W]

   .. py:method:: srgb_to_linear(srgb)

      sRGB to linear RGB.

      :param srgb: sRGB image [3, H, W]
      :return: Linear RGB [3, H, W]

   .. py:method:: linear_to_srgb(linear)

      Linear RGB to sRGB.

      :param linear: Linear RGB [3, H, W]
      :return: sRGB image [3, H, W]

Sharpen
^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.Sharpen(strength=0.3, radius=1.0, device='cuda')

   Image sharpening.

   :param strength: Sharpening strength
   :param radius: Sharpening radius
   :param device: Device

   .. py:method:: forward(rgb)

      Sharpen image.

      :param rgb: RGB image [3, H, W]
      :return: Sharpened image [3, H, W]

AntiAliasing
^^^^^^^^^^^^

.. py:class:: deeplens.sensor.isp_modules.AntiAliasing(sigma=0.5, kernel_size=5, device='cuda')

   Anti-aliasing filter.

   :param sigma: Gaussian blur sigma
   :param kernel_size: Filter kernel size
   :param device: Device

   .. py:method:: forward(image)

      Apply anti-aliasing.

      :param image: Input image
      :return: Filtered image

Examples
--------

Basic Sensor Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor import RGBSensor
    
    # Create sensor
    sensor = RGBSensor(
        resolution=(1920, 1080),
        pixel_size=4.0e-3,
        device='cuda'
    )
    
    # Capture image
    irradiance = get_irradiance()  # From lens system
    raw = sensor.capture(irradiance, exposure_time=0.01, iso=100)

Complete ISP Pipeline
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor import RGBSensor, ISP
    
    # Create sensor and ISP
    sensor = RGBSensor(resolution=(1920, 1080))
    isp = ISP(
        demosaic_method='malvar',
        white_balance=True,
        gamma_correction=True
    )
    
    # Capture and process
    raw = sensor.capture(irradiance, exposure_time=0.01)
    rgb = isp(raw)

Custom ISP Pipeline
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor.isp_modules import *
    import torch.nn as nn
    
    class MyISP(nn.Module):
        def __init__(self):
            super().__init__()
            self.black_level = BlackLevel(level=64)
            self.wb = WhiteBalance(method='gray_world')
            self.demosaic = Demosaic(method='malvar')
            self.ccm = ColorMatrix()
            self.gamma = GammaCorrection(gamma=2.2)
        
        def forward(self, raw):
            x = self.black_level(raw)
            x = self.wb(x)
            x = self.demosaic(x)
            x = self.ccm(x)
            x = self.gamma(x)
            return x
    
    # Use custom ISP
    my_isp = MyISP()
    rgb = my_isp(raw_image)

Camera with Sensor
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import Camera, GeoLens
    from deeplens.sensor import RGBSensor, ISP
    
    # Create components
    lens = GeoLens(filename='lens.json')
    sensor = RGBSensor(resolution=(1920, 1080))
    isp = ISP()
    
    # Create camera
    camera = Camera(lens=lens, sensor=sensor, isp=isp)
    
    # Capture image
    img = camera.capture(scene, depth=1000, exposure_time=0.01)

Noise Simulation
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Enable all noise sources
    sensor = RGBSensor(
        resolution=(1920, 1080),
        enable_shot_noise=True,
        enable_dark_noise=True,
        enable_read_noise=True
    )
    
    # Capture with noise
    raw_noisy = sensor.capture(irradiance, exposure_time=0.01)
    
    # Capture without noise
    sensor.enable_shot_noise = False
    sensor.enable_dark_noise = False
    sensor.enable_read_noise = False
    raw_clean = sensor.capture(irradiance, exposure_time=0.01)

White Balance Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor.isp_modules import WhiteBalance
    
    wb = WhiteBalance(method='gray_world')
    
    # Estimate gains from image
    gains = wb.estimate_gains(raw_image)
    print(f"WB gains: R={gains[0]:.2f}, G={gains[1]:.2f}, B={gains[2]:.2f}")
    
    # Apply white balance
    balanced = wb(raw_image)

See Also
--------

* :doc:`../user_guide/sensors` - Detailed sensor guide
* :doc:`lens` - Lens API for complete imaging simulation
* :doc:`../tutorials` - Tutorials and workflows


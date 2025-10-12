Sensors and ISP
===============

DeepLens provides comprehensive sensor simulation and image signal processing (ISP) pipelines for realistic image capture.

Sensor Types
------------

RGB Sensor
^^^^^^^^^^

Standard Bayer pattern RGB sensor with color filter array.

.. code-block:: python

    from deeplens.sensor import RGBSensor
    
    sensor = RGBSensor(
        resolution=(1920, 1080),    # Width x Height
        pixel_size=4.0e-3,          # Pixel size [mm], 4 micrometers
        bit_depth=12,               # ADC bit depth
        qe=0.6,                     # Quantum efficiency
        dark_current=0.01,          # Dark current [e-/s]
        read_noise=2.0,             # Read noise [e-]
        full_well=10000,            # Full well capacity [e-]
        device='cuda'
    )

**Bayer Pattern:**

The sensor uses a standard RGGB Bayer pattern::

    R G R G R G
    G B G B G B
    R G R G R G
    G B G B G B

Mono Sensor
^^^^^^^^^^^

Monochrome sensor without color filter array.

.. code-block:: python

    from deeplens.sensor import MonoSensor
    
    sensor = MonoSensor(
        resolution=(2048, 2048),
        pixel_size=3.45e-3,
        bit_depth=14,
        qe=0.7,
        device='cuda'
    )

Event Sensor
^^^^^^^^^^^^

Event-based sensor (DVS) for high-speed applications.

.. code-block:: python

    from deeplens.sensor import EventSensor
    
    sensor = EventSensor(
        resolution=(640, 480),
        pixel_size=18.5e-3,
        threshold=0.1,              # Contrast threshold
        refractory_period=1e-3,     # Refractory period [s]
        device='cuda'
    )

Sensor Properties
-----------------

Key Parameters
^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Parameter
     - Description
     - Typical Range
   * - **pixel_size**
     - Physical pixel size
     - 1-20 μm
   * - **bit_depth**
     - ADC resolution
     - 8-16 bits
   * - **qe**
     - Quantum efficiency
     - 0.3-0.9
   * - **dark_current**
     - Dark current noise
     - 0.001-1 e-/s
   * - **read_noise**
     - Read noise
     - 1-10 e-
   * - **full_well**
     - Well capacity
     - 1k-100k e-

Sensor Characteristics
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Print sensor specifications
    print(f"Sensor size: {sensor.sensor_size} mm")
    print(f"Resolution: {sensor.resolution} pixels")
    print(f"Pixel pitch: {sensor.pixel_size*1000:.2f} μm")
    print(f"Sensor diagonal: {sensor.diagonal:.2f} mm")
    print(f"Crop factor: {sensor.crop_factor:.2f}")

Image Capture
-------------

Basic Capture
^^^^^^^^^^^^^

.. code-block:: python

    # Capture image from optical field
    # Input: irradiance on sensor [W/m^2]
    irradiance = lens.get_irradiance()  # [H, W, 3] or [H, W]
    
    # Capture with sensor
    raw_image = sensor.capture(
        irradiance=irradiance,
        exposure_time=0.01,  # Exposure time [s]
        iso=100              # ISO setting
    )
    
    # Output: Raw sensor data with Bayer pattern

Noise Models
^^^^^^^^^^^^

DeepLens simulates realistic sensor noise:

1. **Shot Noise**: Photon counting noise (Poisson)
2. **Dark Current Noise**: Thermal electrons
3. **Read Noise**: Electronic readout noise
4. **Fixed Pattern Noise**: Pixel-to-pixel variation (optional)

.. code-block:: python

    # Enable/disable noise components
    sensor = RGBSensor(
        resolution=(1920, 1080),
        enable_shot_noise=True,
        enable_dark_noise=True,
        enable_read_noise=True,
        enable_fpn=False,
        device='cuda'
    )

Image Signal Processing (ISP)
------------------------------

Complete ISP Pipeline
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor import ISP
    
    # Create ISP pipeline
    isp = ISP(
        demosaic_method='bilinear',    # or 'malvar', 'menon'
        white_balance=True,
        color_correction=True,
        gamma_correction=True,
        denoise=True,
        sharpen=False,
        device='cuda'
    )
    
    # Process raw image
    rgb_image = isp(raw_image)

ISP Modules
-----------

Black Level Correction
^^^^^^^^^^^^^^^^^^^^^^

Remove sensor pedestal:

.. code-block:: python

    from deeplens.sensor.isp_modules import BlackLevel
    
    black_level = BlackLevel(level=64)  # For 12-bit sensor
    corrected = black_level(raw_image)

Lens Shading Correction
^^^^^^^^^^^^^^^^^^^^^^^^

Correct vignetting and non-uniform illumination:

.. code-block:: python

    from deeplens.sensor.isp_modules import LensShadingCorrection
    
    lsc = LensShadingCorrection(
        resolution=(1920, 1080),
        center=[960, 540],
        falloff=0.3,
        device='cuda'
    )
    corrected = lsc(raw_image)

Dead Pixel Correction
^^^^^^^^^^^^^^^^^^^^^^

Remove hot and dead pixels:

.. code-block:: python

    from deeplens.sensor.isp_modules import DeadPixelCorrection
    
    dpc = DeadPixelCorrection(
        threshold=0.1,
        method='median'
    )
    corrected = dpc(raw_image)

White Balance
^^^^^^^^^^^^^

Color temperature correction:

.. code-block:: python

    from deeplens.sensor.isp_modules import WhiteBalance
    
    wb = WhiteBalance(
        method='gray_world',  # or 'white_patch', 'manual'
        gains=[1.5, 1.0, 1.8]  # R, G, B gains (for manual mode)
    )
    balanced = wb(raw_image)

Demosaicing
^^^^^^^^^^^

Convert Bayer pattern to RGB:

.. code-block:: python

    from deeplens.sensor.isp_modules import Demosaic
    
    demosaic = Demosaic(
        method='bilinear'  # or 'malvar', 'menon', 'ahd'
    )
    rgb = demosaic(raw_image)

**Available Methods:**

* ``bilinear``: Fast, simple interpolation
* ``malvar``: Edge-aware interpolation
* ``menon``: High-quality, edge-directed
* ``ahd``: Adaptive homogeneity-directed

Color Correction
^^^^^^^^^^^^^^^^

Apply color correction matrix:

.. code-block:: python

    from deeplens.sensor.isp_modules import ColorMatrix
    
    ccm = ColorMatrix(
        matrix=torch.tensor([
            [1.5, -0.3, -0.2],
            [-0.2, 1.3, -0.1],
            [-0.1, -0.4, 1.5]
        ])
    )
    corrected = ccm(rgb_image)

Gamma Correction
^^^^^^^^^^^^^^^^

Apply gamma curve for display:

.. code-block:: python

    from deeplens.sensor.isp_modules import GammaCorrection
    
    gamma = GammaCorrection(
        gamma=2.2,  # Standard sRGB gamma
        method='power'  # or 'srgb', 'log'
    )
    gamma_corrected = gamma(linear_rgb)

Denoising
^^^^^^^^^

Reduce noise in images:

.. code-block:: python

    from deeplens.sensor.isp_modules import Denoise
    
    denoise = Denoise(
        method='bilateral',  # or 'nlm', 'bm3d'
        strength=0.5
    )
    denoised = denoise(rgb_image)

Color Space Conversion
^^^^^^^^^^^^^^^^^^^^^^

Convert between color spaces:

.. code-block:: python

    from deeplens.sensor.isp_modules import ColorSpace
    
    converter = ColorSpace()
    
    # RGB to YUV
    yuv = converter.rgb_to_yuv(rgb_image)
    
    # RGB to HSV
    hsv = converter.rgb_to_hsv(rgb_image)
    
    # sRGB to linear RGB
    linear = converter.srgb_to_linear(srgb_image)

Sharpening
^^^^^^^^^^

Enhance image sharpness:

.. code-block:: python

    from deeplens.sensor.isp_modules import Sharpen
    
    sharpen = Sharpen(
        strength=0.3,
        radius=1.0
    )
    sharpened = sharpen(rgb_image)

Anti-Aliasing
^^^^^^^^^^^^^

Pre-processing anti-aliasing filter:

.. code-block:: python

    from deeplens.sensor.isp_modules import AntiAliasing
    
    aa = AntiAliasing(
        sigma=0.5,
        kernel_size=5
    )
    filtered = aa(image)

Custom ISP Pipeline
-------------------

Build custom ISP pipelines:

.. code-block:: python

    from deeplens.sensor.isp_modules import *
    
    class CustomISP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.black_level = BlackLevel(level=64)
            self.lsc = LensShadingCorrection(resolution=(1920, 1080))
            self.dpc = DeadPixelCorrection()
            self.wb = WhiteBalance(method='gray_world')
            self.demosaic = Demosaic(method='malvar')
            self.ccm = ColorMatrix()
            self.gamma = GammaCorrection(gamma=2.2)
            self.denoise = Denoise(method='bilateral')
            
        def forward(self, raw):
            x = self.black_level(raw)
            x = self.lsc(x)
            x = self.dpc(x)
            x = self.wb(x)
            x = self.demosaic(x)
            x = self.ccm(x)
            x = self.denoise(x)
            x = self.gamma(x)
            return x
    
    # Use custom ISP
    custom_isp = CustomISP()
    output = custom_isp(raw_image)

Camera System
-------------

Combining Lens and Sensor
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import Camera, GeoLens
    from deeplens.sensor import RGBSensor, ISP
    
    # Create components
    lens = GeoLens(filename='./datasets/lenses/camera/ef50mm_f1.8.json')
    sensor = RGBSensor(resolution=(1920, 1080), pixel_size=4.0e-3)
    isp = ISP()
    
    # Create camera
    camera = Camera(
        lens=lens,
        sensor=sensor,
        isp=isp,
        device='cuda'
    )

End-to-End Capture
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Load scene
    import torch
    from PIL import Image
    import torchvision.transforms as T
    
    scene = Image.open('./datasets/bird.png')
    scene_tensor = T.ToTensor()(scene).unsqueeze(0).cuda()
    
    # Capture image
    captured_image = camera.capture(
        scene=scene_tensor,
        depth=1000.0,           # Object distance [mm]
        exposure_time=0.01,     # 10ms
        iso=100,
        auto_focus=True
    )
    
    # Save result
    from torchvision.utils import save_image
    save_image(captured_image, 'captured.png')

Sensor Calibration
------------------

Flat Field Correction
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Capture flat field image
    flat_field = sensor.capture(uniform_illumination, exposure_time=0.01)
    
    # Create correction map
    correction_map = flat_field.mean() / (flat_field + 1e-6)
    
    # Apply correction
    corrected_image = raw_image * correction_map

Dark Frame Subtraction
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Capture dark frame (no light)
    dark_frame = sensor.capture_dark(exposure_time=0.01)
    
    # Subtract from image
    corrected = raw_image - dark_frame

Sensor Response Curve
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Measure sensor response
    exposures = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    responses = []
    
    for exp in exposures:
        raw = sensor.capture(irradiance, exposure_time=exp)
        responses.append(raw.mean())
    
    # Plot response curve
    import matplotlib.pyplot as plt
    plt.plot(exposures, responses)
    plt.xlabel('Exposure Time [s]')
    plt.ylabel('Sensor Response [DN]')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

Sensor Formats
--------------

Common Sensor Sizes
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Format
     - Width (mm)
     - Height (mm)
     - Diagonal (mm)
   * - Full Frame
     - 36.0
     - 24.0
     - 43.3
   * - APS-C (Canon)
     - 22.2
     - 14.8
     - 26.7
   * - APS-C (Nikon)
     - 23.5
     - 15.6
     - 28.2
   * - Micro 4/3
     - 17.3
     - 13.0
     - 21.6
   * - 1"
     - 13.2
     - 8.8
     - 15.9
   * - 1/1.7"
     - 7.6
     - 5.7
     - 9.5
   * - 1/2.3"
     - 6.2
     - 4.6
     - 7.7

Creating Standard Sensors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Full frame sensor
    sensor_ff = RGBSensor(
        resolution=(6000, 4000),
        sensor_size=(36.0, 24.0),
        pixel_size=6.0e-3
    )
    
    # APS-C sensor
    sensor_apsc = RGBSensor(
        resolution=(6000, 4000),
        sensor_size=(23.5, 15.6),
        pixel_size=3.9e-3
    )
    
    # Smartphone sensor
    sensor_phone = RGBSensor(
        resolution=(4000, 3000),
        sensor_size=(6.2, 4.6),
        pixel_size=1.55e-3
    )

Performance Optimization
------------------------

GPU Acceleration
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Ensure GPU usage
    sensor = RGBSensor(resolution=(1920, 1080), device='cuda')
    
    # Pre-allocate buffers
    sensor.allocate_buffers()
    
    # Batch processing
    batch_size = 8
    raw_batch = sensor.capture_batch(irradiance_batch, exposure_time=0.01)

Memory Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Process in tiles for large images
    def process_tiled(image, tile_size=512):
        H, W = image.shape[-2:]
        output = torch.zeros_like(image)
        
        for i in range(0, H, tile_size):
            for j in range(0, W, tile_size):
                tile = image[..., i:i+tile_size, j:j+tile_size]
                output[..., i:i+tile_size, j:j+tile_size] = isp(tile)
        
        return output

Best Practices
--------------

Sensor Selection
^^^^^^^^^^^^^^^^

1. **Pixel Size**: Larger pixels → better SNR, smaller pixels → higher resolution
2. **Bit Depth**: 12-14 bits sufficient for most applications
3. **Quantum Efficiency**: Higher QE → better low-light performance
4. **Full Well**: Larger well → higher dynamic range

ISP Pipeline Design
^^^^^^^^^^^^^^^^^^^

1. **Order Matters**: Apply corrections in proper sequence
2. **Preserve Data**: Use linear processing until final gamma
3. **White Balance**: Apply early in pipeline for best color
4. **Denoise**: Balance noise reduction vs. detail preservation

Simulation Accuracy
^^^^^^^^^^^^^^^^^^^

1. **Calibrate Sensor**: Use measured parameters when available
2. **Validate Noise**: Compare noise statistics with real sensor
3. **Color Accuracy**: Measure and apply correct CCM
4. **Test Cases**: Validate against real camera captures

Next Steps
----------

* Combine with :doc:`lens_systems` for complete imaging simulation
* Learn about :doc:`neural_networks` for computational photography
* See :doc:`../examples/image_simulation` for complete examples


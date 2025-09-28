# Image Signal Processing (ISP) Modules

This directory contains a collection of PyTorch modules that implement various stages of a typical Image Signal Processing (ISP) pipeline. These modules are designed to process raw sensor data and convert it into a viewable image format.

## Modules

The following ISP modules are available:

- [Image Signal Processing (ISP) Modules](#image-signal-processing-isp-modules)
  - [Modules](#modules)
    - [Anti-Aliasing Filter](#anti-aliasing-filter)
    - [Black Level Compensation](#black-level-compensation)
    - [Color Correction Matrix](#color-correction-matrix)
    - [Color Space Conversion](#color-space-conversion)
    - [Dead Pixel Correction](#dead-pixel-correction)
    - [Demosaic](#demosaic)
    - [Denoise](#denoise)
    - [Gamma Correction](#gamma-correction)
    - [Lens Shading Correction](#lens-shading-correction)
    - [Auto White Balance](#auto-white-balance)

### Anti-Aliasing Filter

**File:** `anti_alising.py`
**Class:** `AntiAliasingFilter`

This module is intended to apply an anti-aliasing filter to reduce moiré patterns in the image.

**Note:** This module is not fully implemented or tested yet.

### Black Level Compensation

**File:** `black_level.py`
**Class:** `BlackLevelCompensation`

This module corrects for the sensor's black level. It subtracts the black level value from the raw sensor data and normalizes the image to a floating-point representation. It also provides a `reverse` method to undo the operation.

### Color Correction Matrix

**File:** `color_matrix.py`
**Class:** `ColorCorrectionMatrix`

This module applies a color correction matrix (CCM) to the image to correct for color inaccuracies of the sensor. It converts the RGB image to the sensor's color space. A `reverse` method is available to convert from the sensor's color space back to RGB.

### Color Space Conversion

**File:** `color_space.py`
**Class:** `ColorSpaceConversion`

This module handles conversions between different color spaces. It currently supports conversion between RGB and YCrCb.

### Dead Pixel Correction

**File:** `dead_pixel.py`
**Class:** `DeadPixelCorrection`

This module identifies and corrects dead pixels in the raw sensor data. It uses a median filter to replace pixels that deviate significantly from their neighbors.

### Demosaic

**File:** `demosaic.py`
**Class:** `Demosaic`

This module converts a single-channel Bayer-patterned image from the sensor into a three-channel RGB image. It supports both `bilinear` and a `3x3` kernel-based interpolation method. It also has a `reverse` method to convert an RGB image back to a Bayer pattern.

### Denoise

**File:** `denoise.py`
**Class:** `Denoise`

This module reduces noise in the image. It supports `gaussian` and `median` filtering methods.

### Gamma Correction

**File:** `gamma_correction.py`
**Class:** `GammaCorrection`

This module applies gamma correction to adjust the luminance of the image, making it more suitable for display. It includes a `reverse` method to invert the gamma correction.

### Lens Shading Correction

**File:** `lens_shading.py`
**Class:** `LensShadingCorrection`

This module is intended to correct for vignetting, which is the reduction of an image's brightness or saturation at the periphery compared to the image center.

**Note:** This module is not yet implemented.

### Auto White Balance

**File:** `white_balance.py`
**Class:** `AutoWhiteBalance`

This module corrects the color balance of the image. It can be applied to either Bayer or RGB images. It supports the `gray_world` algorithm for automatic white balancing and also allows for `manual` gain adjustments. A `reverse` method is provided to undo the white balance operation.

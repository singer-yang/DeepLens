Image Simulation
================

Photorealistic image rendering through optical systems.

Overview
--------

DeepLens provides accurate image simulation including:

* **Optical aberrations**: Chromatic, spherical, coma, astigmatism, etc.
* **Defocus blur**: Depth-dependent blur
* **Diffraction**: Wave optics effects
* **Sensor effects**: Noise, color filter array, ISP pipeline

Basic Image Rendering
----------------------

Single Depth
^^^^^^^^^^^^

.. code-block:: python

    import torch
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.utils import save_image
    from deeplens import GeoLens
    from deeplens.basics import DEPTH, SPP_RENDER
    
    # Load lens
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Load image (must match sensor resolution for ray_tracing method)
    img = Image.open('./datasets/bird.png')
    img_tensor = T.ToTensor()(img).unsqueeze(0).cuda()
    
    # Resize to match sensor resolution
    img_tensor = T.functional.resize(img_tensor, lens.sensor_res[::-1])
    
    # Render through lens
    # Methods: 'ray_tracing' (accurate), 'psf_map' (efficient), 'psf_patch' (single PSF)
    img_rendered = lens.render(
        img_tensor,
        depth=DEPTH,           # Object depth (-20000.0 mm default)
        method='ray_tracing',  # Ray tracing rendering
        spp=SPP_RENDER         # Samples per pixel (32 default)
    )
    
    # Save result
    save_image(img_rendered, 'rendered.png')

Multiple Depths
^^^^^^^^^^^^^^^

Render objects at different distances:

.. code-block:: python

    from deeplens.basics import SPP_RENDER
    
    # Depths are negative (object in front of lens)
    depths = [-500, -1000, -2000, -5000, -10000]
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(depths), figsize=(20, 4))
    
    for i, depth in enumerate(depths):
        img_rendered = lens.render(
            img_tensor, 
            depth=depth, 
            method='ray_tracing',
            spp=SPP_RENDER
        )
        axes[i].imshow(img_rendered[0].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[i].set_title(f'{abs(depth)} mm')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_series.png', dpi=150)

Depth Map Rendering
-------------------

For scenes with varying depth, use PSF map convolution:

.. code-block:: python

    from PIL import Image
    import torch
    import torchvision.transforms as T
    from deeplens.basics import PSF_KS, SPP_PSF
    
    # Load RGB image
    img_rgb = Image.open('./datasets/edof/rgb.png')
    rgb_tensor = T.ToTensor()(img_rgb).unsqueeze(0).cuda()
    
    # Resize to sensor resolution
    rgb_tensor = T.functional.resize(rgb_tensor, lens.sensor_res[::-1])
    
    # Use PSF map for spatially-varying blur
    # PSF map renders with field-dependent PSFs across the image
    img_rendered = lens.render(
        rgb_tensor,
        depth=-2000.0,       # Object depth
        method='psf_map',    # Use PSF map convolution
        psf_grid=(10, 10),   # Grid of PSFs across the field
        psf_ks=PSF_KS        # PSF kernel size (128 default)
    )
    
    save_image(img_rendered, 'psf_map_rendered.png')

PSF Map vs Ray Tracing
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Two main rendering methods:
    
    # 1. ray_tracing: Accurate but slower
    #    - Traces rays from sensor through lens to object
    #    - Handles all aberrations and vignetting
    #    - Best for evaluation
    img_rt = lens.render(img, depth=-2000.0, method='ray_tracing', spp=64)
    
    # 2. psf_map: Efficient for training
    #    - Computes PSF at grid points and convolves
    #    - Field-dependent blur approximation
    #    - Differentiable and memory-efficient
    img_psf = lens.render(img, depth=-2000.0, method='psf_map', psf_grid=(7, 7))

High-Quality Rendering
----------------------

Wave Optics
^^^^^^^^^^^

For accurate diffraction simulation, use coherent PSF:

.. code-block:: python

    import torch
    from deeplens.basics import PSF_KS, SPP_COHERENT
    
    # For wave optics PSF, use psf_coherent() method
    # Requires double precision for accurate phase computation
    torch.set_default_dtype(torch.float64)
    lens.astype(torch.float64)
    
    # Compute coherent (wave optics) PSF
    point = torch.tensor([0.0, 0.0, -10000.0])
    psf_wave = lens.psf_coherent(
        points=point,
        ks=PSF_KS,
        wvln=0.550,
        spp=SPP_COHERENT  # ~16.7M rays for accurate phase
    )

Wave optics accounts for:

* Diffraction at aperture
* Interference effects
* Wavelength-dependent PSF structure

Multi-Wavelength
^^^^^^^^^^^^^^^^

The render() method automatically handles RGB wavelengths:

.. code-block:: python

    from deeplens.basics import WAVE_RGB, SPP_RENDER
    
    # render() uses WAVE_RGB = [0.656, 0.588, 0.486] um for R, G, B channels
    # Each channel is rendered with its corresponding wavelength
    img_rendered = lens.render(
        img_tensor,  # RGB image [B, 3, H, W]
        depth=-2000.0,
        method='ray_tracing',
        spp=SPP_RENDER
    )
    
    # For custom wavelength PSFs, compute directly:
    import torch
    from deeplens.basics import PSF_KS, SPP_PSF
    
    wavelengths = WAVE_RGB  # [0.656, 0.588, 0.486]
    point = torch.tensor([0.0, 0.0, -2000.0])
    
    psfs = []
    for wvln in wavelengths:
        psf = lens.psf(points=point, ks=PSF_KS, wvln=wvln, spp=SPP_PSF)
        psfs.append(psf)
    
    # Stack to get RGB PSF [3, ks, ks]
    psf_rgb = torch.stack(psfs, dim=0)

Field-Dependent PSF
^^^^^^^^^^^^^^^^^^^

Use PSF map for spatially-varying blur:

.. code-block:: python

    from deeplens.basics import PSF_KS, SPP_PSF, DEPTH
    import torch
    
    # Method 1: Use render() with psf_map method (recommended)
    img_rendered = lens.render(
        img_tensor,
        depth=DEPTH,
        method='psf_map',
        psf_grid=(7, 7),    # 7x7 grid of PSFs
        psf_ks=PSF_KS       # PSF kernel size
    )
    
    # Method 2: Compute PSF map directly for visualization
    psf_map = lens.psf_map(
        depth=DEPTH,
        grid=(7, 7),        # Grid size (grid_w, grid_h)
        ks=PSF_KS,
        spp=SPP_PSF,
        recenter=True       # Recenter PSF using chief ray
    )
    # psf_map shape: [grid_h, grid_w, 1, ks, ks]
    
    # Visualize PSF map
    lens.draw_psf_map(save_name='./psf_map.png', depth=DEPTH, show=False)

Complete Camera Simulation
---------------------------

Including Sensor and ISP
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.sensor import RGBSensor
    
    # RGBSensor loads configuration from JSON file
    # Config includes: resolution, bit depth, noise parameters, ISP settings
    sensor = RGBSensor(sensor_file='./datasets/sensors/imx586.json')
    
    # The sensor includes:
    # - Noise simulation (shot noise, read noise)
    # - ISP pipeline (demosaicing, white balance, color correction, gamma)
    
    # Render image through lens first
    from deeplens.basics import DEPTH, SPP_RENDER
    
    img_rendered = lens.render(
        img_tensor, 
        depth=DEPTH, 
        method='ray_tracing',
        spp=SPP_RENDER
    )
    
    # Convert to n-bit raw space (simulate sensor response)
    # img_rendered is in [0, 1] linear space
    bit = sensor.bit
    black_level = sensor.black_level
    img_nbit = img_rendered * (2**bit - 1 - black_level) + black_level
    
    # Apply sensor noise and ISP
    iso = 100
    img_rgb = sensor.forward(img_nbit, iso=iso)
    
    save_image(img_rgb, 'camera_captured.png')

Bokeh Effects
-------------

Circular Bokeh
^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.basics import SPP_RENDER
    
    # Defocus effects depend on object distance
    # Objects far from focus distance have larger blur
    
    # Refocus lens to specific distance
    lens.refocus(foc_dist=-1000.0)  # Focus at 1m
    
    # Render at different depths
    img_focus = lens.render(img_tensor, depth=-1000.0, method='ray_tracing', spp=SPP_RENDER)
    img_defocus = lens.render(img_tensor, depth=-5000.0, method='ray_tracing', spp=SPP_RENDER)
    
    # Compare
    from torchvision.utils import save_image
    save_image(torch.cat([img_focus, img_defocus], dim=0), 'bokeh_comparison.png', nrow=2)

Shaped Aperture Bokeh
^^^^^^^^^^^^^^^^^^^^^

Bokeh shape depends on aperture:

.. code-block:: python

    from deeplens.optics.geometric_surface import Aperture
    
    # Find aperture in lens
    for i, surf in enumerate(lens.surfaces):
        if isinstance(surf, Aperture):
            # Aperture radius controls bokeh size
            # Smaller aperture (higher f-number) = sharper but dimmer
            print(f"Aperture at surface {i}, radius: {surf.r}")
    
    # Change f-number to control bokeh size
    lens.set_fnum(fnum=2.8)  # Larger aperture = more bokeh
    
    # Render with new aperture
    from deeplens.basics import SPP_RENDER
    img_rendered = lens.render(img_tensor, depth=-5000.0, method='ray_tracing', spp=SPP_RENDER)

Computational Photography
--------------------------

Extended Depth of Field (EDoF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using lens with cubic phase element:

.. code-block:: python

    from deeplens import GeoLens
    from deeplens.basics import SPP_RENDER
    
    # Load lens with cubic phase element for EDoF
    # Cubic phase creates depth-invariant PSF
    lens_edof = GeoLens(
        filename='./datasets/lenses/camera/edof_cubic.json',
        device='cuda'
    )
    
    # Render at multiple depths - PSF is similar across depths
    depths = [-500.0, -1000.0, -2000.0, -5000.0]
    
    for depth in depths:
        img_d = lens_edof.render(
            img_tensor, 
            depth=depth, 
            method='ray_tracing',
            spp=SPP_RENDER
        )
        # All depths produce similar blur that can be deconvolved
    
    # Use network to restore sharp image from any depth

Focus Stacking
^^^^^^^^^^^^^^

.. code-block:: python

    def focus_stack(img, depths, lens):
        """Create focus-stacked image."""
        images = []
        for depth in depths:
            img_d = lens.render(img, depth=depth, spp=512)
            images.append(img_d)
        
        # Combine using maximum gradient
        stack = torch.stack(images, dim=0)
        
        # Simple max sharpness approach
        gradients = torch.abs(
            stack[:, :, :, 1:] - stack[:, :, :, :-1]
        ).sum(dim=2)
        
        indices = gradients.argmax(dim=0)
        # ... complex selection logic
        
        return focused_img
    
    img_stacked = focus_stack(
        img_tensor,
        depths=torch.linspace(500, 2000, 20),
        lens=lens
    )

Light Field Rendering
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Render from multiple viewpoints
    viewpoints = 7
    light_field = []
    
    for i in range(viewpoints):
        for j in range(viewpoints):
            # Offset aperture center
            offset_x = (i - viewpoints//2) * 0.5
            offset_y = (j - viewpoints//2) * 0.5
            
            # Render with offset
            img_view = lens.render_with_offset(
                img_tensor,
                depth=1000,
                offset=[offset_x, offset_y]
            )
            light_field.append(img_view)
    
    # Save light field
    light_field_tensor = torch.cat(light_field, dim=0)
    save_image(light_field_tensor, 'light_field.png', nrow=viewpoints)

Performance Optimization
------------------------

Tile-Based Rendering
^^^^^^^^^^^^^^^^^^^^

For large images:

.. code-block:: python

    def render_tiled(img, depth, lens, tile_size=256, overlap=32):
        """Memory-efficient tile-based rendering."""
        B, C, H, W = img.shape
        output = torch.zeros_like(img)
        
        # Calculate PSF once
        psf = lens.psf(depth=depth, spp=1024)
        
        for i in range(0, H, tile_size - overlap):
            for j in range(0, W, tile_size - overlap):
                # Extract tile with overlap
                i1, i2 = i, min(i + tile_size, H)
                j1, j2 = j, min(j + tile_size, W)
                tile = img[:, :, i1:i2, j1:j2]
                
                # Render tile
                tile_rendered = lens.convolve_with_psf(tile, psf)
                
                # Blend into output
                output[:, :, i1:i2, j1:j2] = tile_rendered
        
        return output

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Process multiple images in batch
    image_paths = ['img1.png', 'img2.png', 'img3.png']
    images = []
    
    for path in image_paths:
        img = Image.open(path)
        images.append(T.ToTensor()(img))
    
    # Batch render
    batch = torch.stack(images).cuda()
    rendered_batch = lens.render(batch, depth=1000, spp=512)
    
    # Save results
    for i, img in enumerate(rendered_batch):
        save_image(img, f'rendered_{i}.png')

Quality Assessment
------------------

Compare with Ground Truth
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.utils import batch_psnr, batch_ssim
    from deeplens.basics import DEPTH, SPP_RENDER
    
    # Reference (sharp) image
    img_ref = img_tensor
    
    # Simulated image
    img_sim = lens.render(
        img_tensor, 
        depth=DEPTH, 
        method='ray_tracing',
        spp=SPP_RENDER
    )
    
    # Metrics
    psnr = batch_psnr(img_sim, img_ref)
    ssim = batch_ssim(img_sim, img_ref)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

Validation Against Real Camera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Capture with real camera
    img_real = Image.open('real_capture.jpg')
    img_real_tensor = T.ToTensor()(img_real).unsqueeze(0).cuda()
    
    # Simulate with DeepLens
    img_sim = camera.capture(scene, depth=1000, exposure_time=0.01)
    
    # Compare
    plot_comparison([img_real_tensor, img_sim], ['Real', 'Simulated'])

Tips and Best Practices
------------------------

1. **SPP Selection**: 32 (SPP_RENDER) for training, 64+ for evaluation
2. **Method**: Use 'ray_tracing' for accuracy, 'psf_map' for efficiency
3. **Memory**: Image resolution must match sensor_res for ray_tracing
4. **Depth**: Negative values (object in front of lens), typically -500 to -20000 mm
5. **Field Variation**: Use 'psf_map' method with psf_grid for wide-angle lenses
6. **Wavelength**: render() automatically uses RGB wavelengths for 3-channel images
7. **Validation**: Use analysis_rendering() for comprehensive evaluation with metrics

See Also
--------

* :doc:`../tutorials` - Step-by-step tutorials
* :doc:`../user_guide/lens_systems` - Lens system details
* :doc:`../user_guide/sensors` - Sensor simulation
* Example script: ``7_image_simulation.py``


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
    
    # Load lens
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Load image
    img = Image.open('./datasets/bird.png')
    img_tensor = T.ToTensor()(img).unsqueeze(0).cuda()
    
    # Render through lens
    img_rendered = lens.render(
        img_tensor,
        depth=1000.0,  # 1 meter
        spp=512,       # Samples per pixel
        method='fft'   # FFT-based convolution
    )
    
    # Save result
    save_image(img_rendered, 'rendered.png')

Multiple Depths
^^^^^^^^^^^^^^^

Render objects at different distances:

.. code-block:: python

    depths = [500, 1000, 2000, 5000, 10000]
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(depths), figsize=(20, 4))
    
    for i, depth in enumerate(depths):
        img_rendered = lens.render(img_tensor, depth=depth, spp=512)
        axes[i].imshow(img_rendered[0].permute(1, 2, 0).cpu())
        axes[i].set_title(f'{depth} mm')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_series.png', dpi=150)

Depth Map Rendering
-------------------

Render scenes with varying depth (extended depth of field):

.. code-block:: python

    from PIL import Image
    import torch
    import torchvision.transforms as T
    
    # Load RGB and depth map
    img_rgb = Image.open('./datasets/edof/rgb.png')
    img_depth = Image.open('./datasets/edof/depth.png')
    
    rgb_tensor = T.ToTensor()(img_rgb).unsqueeze(0).cuda()
    depth_tensor = T.ToTensor()(img_depth).unsqueeze(0).cuda()
    
    # Scale depth to physical units (mm)
    # Assume depth map is normalized [0, 1] → [500mm, 5000mm]
    depth_mm = depth_tensor * 4500 + 500
    
    # Render with spatially-varying depth
    img_rendered = lens.render_depth(
        rgb_tensor,
        depth_mm,
        spp=512
    )
    
    save_image(img_rendered, 'depth_rendered.png')

Implementation
^^^^^^^^^^^^^^

The depth-aware rendering:

.. code-block:: python

    def render_depth(self, img, depth_map, spp=256, tile_size=64):
        """Render image with spatially-varying depth.
        
        Args:
            img: RGB image [B, 3, H, W]
            depth_map: Depth map [B, 1, H, W] in mm
            spp: Samples per pixel
            tile_size: Tile size for memory efficiency
        """
        B, C, H, W = img.shape
        output = torch.zeros_like(img)
        
        # Get unique depth values (quantize for efficiency)
        depths_unique = depth_map.unique()
        
        # Render each depth
        psf_cache = {}
        for depth in depths_unique:
            if depth.item() not in psf_cache:
                psf = self.psf(depth=depth.item(), spp=spp)
                psf_cache[depth.item()] = psf
        
        # Composite based on depth
        for depth in depths_unique:
            mask = (depth_map == depth).float()
            if mask.sum() > 0:
                psf = psf_cache[depth.item()]
                img_blurred = self.convolve_with_psf(img, psf)
                output += img_blurred * mask
        
        return output

High-Quality Rendering
----------------------

Wave Optics
^^^^^^^^^^^

For accurate diffraction simulation:

.. code-block:: python

    # Use wave optics PSF
    img_rendered = lens.render(
        img_tensor,
        depth=1000,
        spp=2048,
        method='fft',
        psf_method='wave'  # Wave optics (default)
    )

This accounts for:

* Diffraction at aperture
* Interference effects
* Wavelength-dependent PSF

Multi-Wavelength
^^^^^^^^^^^^^^^^

Separate rendering for each wavelength:

.. code-block:: python

    wavelengths = [0.486, 0.550, 0.656]  # Blue, green, red
    channels = []
    
    for i, wvln in enumerate(wavelengths):
        # Render single channel
        channel = lens.render(
            img_tensor[:, i:i+1],  # Single channel
            depth=1000,
            spp=1024,
            wavelength=wvln
        )
        channels.append(channel)
    
    # Combine RGB channels
    img_rendered = torch.cat(channels, dim=1)
    save_image(img_rendered, 'chromatic.png')

Field-Dependent PSF
^^^^^^^^^^^^^^^^^^^

Use spatially-varying PSF across the image:

.. code-block:: python

    def render_field_dependent(img, depth, num_tiles=5):
        """Render with field-dependent PSF."""
        B, C, H, W = img.shape
        output = torch.zeros_like(img)
        
        # Divide image into tiles
        tile_h = H // num_tiles
        tile_w = W // num_tiles
        
        for i in range(num_tiles):
            for j in range(num_tiles):
                # Calculate field position for this tile
                field_y = (i - num_tiles/2) / (num_tiles/2)
                field_x = (j - num_tiles/2) / (num_tiles/2)
                
                # Get PSF for this field
                psf = lens.psf(
                    depth=depth,
                    field=[field_x, field_y],
                    spp=1024
                )
                
                # Extract tile
                y1, y2 = i * tile_h, (i+1) * tile_h
                x1, x2 = j * tile_w, (j+1) * tile_w
                tile = img[:, :, y1:y2, x1:x2]
                
                # Render tile
                tile_rendered = convolve_with_psf(tile, psf)
                output[:, :, y1:y2, x1:x2] = tile_rendered
        
        return output
    
    img_rendered = render_field_dependent(img_tensor, depth=1000)

Complete Camera Simulation
---------------------------

Including Sensor and ISP
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import Camera
    from deeplens.sensor import RGBSensor, ISP
    
    # Create camera system
    camera = Camera(
        lens=lens,
        sensor=RGBSensor(
            resolution=(1920, 1080),
            pixel_size=4.0e-3,
            enable_shot_noise=True,
            enable_read_noise=True
        ),
        isp=ISP(
            demosaic_method='malvar',
            white_balance=True,
            gamma_correction=True
        ),
        device='cuda'
    )
    
    # Capture image
    img_captured = camera.capture(
        scene=img_tensor,
        depth=1000,
        exposure_time=0.01,  # 10ms
        iso=100
    )
    
    save_image(img_captured, 'camera_captured.png')

Bokeh Effects
-------------

Circular Bokeh
^^^^^^^^^^^^^^

.. code-block:: python

    # Defocus background while focusing on foreground
    focus_distance = 1000  # Focus at 1m
    
    # Object in focus
    img_focus = lens.render(img_tensor, depth=focus_distance, spp=512)
    
    # Background out of focus
    img_background = lens.render(img_tensor, depth=5000, spp=512)
    
    # Composite
    from deeplens.utils import plot_comparison
    plot_comparison(
        [img_focus, img_background],
        ['In Focus (1m)', 'Out of Focus (5m)']
    )

Shaped Aperture Bokeh
^^^^^^^^^^^^^^^^^^^^^

Create custom bokeh shapes:

.. code-block:: python

    from deeplens.optics import Aperture
    
    # Replace aperture with custom shape
    # Find aperture in lens
    for i, surf in enumerate(lens.surfaces):
        if isinstance(surf, Aperture):
            # Modify aperture shape
            surf.is_square = True  # Square bokeh
            # Or create custom shape
    
    # Render with custom bokeh
    img_rendered = lens.render(img_tensor, depth=5000, spp=1024)

Computational Photography
--------------------------

Extended Depth of Field (EDoF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using cubic phase mask:

.. code-block:: python

    from deeplens.optics import Cubic
    
    # Add cubic phase element
    lens.surfaces.insert(
        aperture_index + 1,
        Cubic(r=float('inf'), d=0.001, alpha=10.0)
    )
    
    # Render at multiple depths
    depths = [500, 1000, 2000, 5000]
    img_edof = torch.zeros_like(img_tensor)
    
    for depth in depths:
        img_d = lens.render(img_tensor, depth=depth, spp=512)
        img_edof += img_d / len(depths)
    
    save_image(img_edof, 'edof.png')

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
    
    # Reference (sharp) image
    img_ref = img_tensor
    
    # Simulated image
    img_sim = lens.render(img_tensor, depth=1000, spp=512)
    
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

1. **SPP Selection**: 256-512 for preview, 1024-4096 for final render
2. **Method**: Use 'fft' for speed, 'conv' for very large PSFs
3. **Memory**: Use tiled rendering for large images (>2K resolution)
4. **Depth Range**: Render at representative depths for your application
5. **Field Variation**: Use field-dependent PSF for wide-angle lenses
6. **Wavelength**: Separate RGB rendering for chromatic aberrations
7. **Validation**: Always validate against reference images or real captures

See Also
--------

* :doc:`../tutorials` - Step-by-step tutorials
* :doc:`../user_guide/lens_systems` - Lens system details
* :doc:`../user_guide/sensors` - Sensor simulation
* Example script: ``7_image_simulation.py``


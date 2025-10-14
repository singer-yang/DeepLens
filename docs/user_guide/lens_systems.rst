Lens Systems
============

DeepLens provides several types of lens models, each suited for different applications and computational requirements.

Overview
--------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Lens Type
     - Description
     - Use Case
   * - **GeoLens**
     - Ray tracing geometric lens
     - High-accuracy simulation, lens design
   * - **DiffracLens**
     - Wave optics diffractive lens
     - Diffractive optical elements
   * - **HybridLens**
     - Hybrid refractive-diffractive
     - DOEs, metasurfaces, hybrid systems
   * - **PSFNetLens**
     - Neural surrogate model
     - Fast PSF prediction, real-time applications
   * - **ParaxialLens**
     - Paraxial approximation
     - Quick prototyping, defocus simulation

GeoLens - Geometric Ray Tracing
--------------------------------

The ``GeoLens`` class implements a fully differentiable ray tracing engine for refractive lens systems.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import GeoLens
    
    # Load from file
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_res=(2000, 2000),
        sensor_size=(8.0, 8.0),  # mm
        device='cuda'
    )
    
    # Create from scratch
    lens = GeoLens(
        sensor_res=(1024, 1024),
        sensor_size=(10.0, 10.0),
        device='cuda'
    )

Supported Surface Types
^^^^^^^^^^^^^^^^^^^^^^^

* **Spheric**: Standard spherical surfaces
* **Aspheric**: Aspheric surfaces with even polynomial terms
* **AsphericNorm**: Normalized aspheric surfaces
* **Plane**: Flat surfaces
* **Aperture**: Aperture stops
* **ThinLens**: Paraxial thin lens approximation
* **Cubic**: Cubic phase surfaces
* **Phase**: General phase surfaces

Material Library
^^^^^^^^^^^^^^^^

GeoLens includes extensive material databases:

* **SCHOTT**: Standard optical glasses
* **CDGM**: Chinese optical glasses
* **PLASTIC**: Optical plastics
* **MISC**: Miscellaneous materials

.. code-block:: python

    from deeplens.optics import Material
    
    # Load material
    material = Material('N-BK7')
    
    # Get refractive index at wavelength (in micrometers)
    n = material.refractive_index(0.550)  # 550 nm = 0.550 μm

Adding Surfaces Manually
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import Spheric, Aspheric, Aperture
    from deeplens.optics import Material
    
    lens = GeoLens(device='cuda')
    
    # Add surfaces
    lens.surfaces.append(Spheric(r=50.0, d=5.0, is_square=False))
    lens.materials.append(Material('N-BK7'))
    
    lens.surfaces.append(Spheric(r=-50.0, d=45.0, is_square=False))
    lens.materials.append(Material('air'))
    
    lens.surfaces.append(Aperture(r=10.0, d=0.0))
    lens.materials.append(Material('air'))
    
    # Finalize lens
    lens.post_computation()

Ray Tracing Methods
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Point source ray tracing
    ray = lens.sample_point_source(depth=1000, num_rays=512)
    ray_out = lens.trace(ray)
    
    # Parallel ray bundle (2D visualization)
    ray = lens.sample_parallel_2D(fov=0.0, num_rays=7)
    ray_out = lens.trace(ray)
    
    # Sample rays from 3D points
    ray = lens.sample_from_points(
        points=[[0.0, 0.0, -1000.0]],
        num_rays=1024
    )
    ray_out = lens.trace(ray)

PSF Calculation
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    
    # Single point PSF (center field, at depth -1000mm)
    points = torch.tensor([[0.0, 0.0, -1000.0]])
    psf = lens.psf(points=points, ks=51, spp=4096)
    
    # Off-axis PSF (normalized coordinates)
    points = torch.tensor([[0.5, 0.3, -1000.0]])  # x, y normalized [-1, 1]
    psf = lens.psf(points=points, ks=51, spp=2048)
    
    # RGB PSF
    psf_rgb = lens.psf_rgb(points=points, ks=51, spp=1024)

Image Rendering
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from torchvision.utils import save_image
    
    # Load image as tensor (must match sensor resolution)
    img = torch.rand(1, 3, 2000, 2000).cuda()
    
    # Render through lens using PSF map
    img_rendered = lens.render(
        img,
        depth=-1000,
        method='psf_map',
        psf_grid=(10, 10),
        psf_ks=51
    )
    
    # Or use ray tracing (more accurate, slower)
    img_rendered = lens.render(
        img,
        depth=-1000,
        method='ray_tracing',
        spp=512
    )
    
    save_image(img_rendered, 'output.png')

DiffractiveLens - Wave Optics
------------------------------

``DiffractiveLens`` implements wave optics for diffractive optical elements.

.. code-block:: python

    from deeplens import DiffractiveLens
    
    lens = DiffractiveLens(
        filename='./datasets/lenses/doe/doe_example.json',
        sensor_res=(2000, 2000),
        sensor_size=(8.0, 8.0),
        device='cuda'
    )

Supported Diffractive Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fresnel**: Fresnel zone plates
* **Binary2**: Binary diffractive surfaces
* **Pixel2D**: Pixelated metasurfaces
* **Zernike**: Zernike polynomial surfaces

HybridLens - Refractive-Diffractive
------------------------------------

``HybridLens`` combines ray tracing and wave optics for accurate simulation of hybrid systems.

.. code-block:: python

    from deeplens import HybridLens
    
    lens = HybridLens(
        filename='./datasets/lenses/hybridlens/hybrid_design.json',
        sensor_res=(2000, 2000),
        sensor_size=(8.0, 8.0),
        device='cuda',
        dtype=torch.float64
    )
    
    # Render image through hybrid lens
    img_rendered = lens.render(img, depth=1000)

Features
^^^^^^^^

* Accurate chromatic aberration simulation
* Support for DOEs and metasurfaces
* Polarization effects (optional)
* Wavelength-dependent diffraction

PSFNetLens - Neural Surrogate
------------------------------

``PSFNetLens`` uses neural networks to predict PSFs, enabling real-time applications.

.. code-block:: python

    from deeplens import PSFNetLens
    
    # Initialize PSFNetLens with lens file
    lens = PSFNetLens(
        lens_path='./datasets/lenses/camera/ef50mm_f1.8.json',
        in_chan=3,
        psf_chan=3,
        model_name='mlp_conv',
        kernel_size=64,
        sensor_res=(3000, 3000)
    )
    
    # Load pre-trained network weights
    lens.load_net('./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth')
    
    # Fast image rendering
    img_rendered = lens.render(img, depth=1000)

Advantages
^^^^^^^^^^

* 100-1000x faster than ray tracing
* Differentiable for end-to-end optimization
* Compact model size (~10MB)
* Supports depth and field variation

Training PSFNet
^^^^^^^^^^^^^^^

To train your own PSFNet model:

.. code-block:: bash

    python 3_psf_net.py

See :doc:`../tutorials` for detailed training instructions.

ParaxialLens - Quick Prototyping
---------------------------------

``ParaxialLens`` implements a paraxial (thin lens) model for rapid prototyping.

.. code-block:: python

    from deeplens import ParaxialLens
    
    lens = ParaxialLens(
        foclen=50.0,              # Focal length in mm
        fnum=2.0,                 # F-number
        sensor_size=(8.0, 8.0),   # Sensor size in mm (W, H)
        sensor_res=(512, 512),    # Sensor resolution in pixels (W, H)
        device='cuda'
    )
    
    # Refocus the lens
    lens.refocus(foc_dist=-2000)
    
    # Defocus blur simulation
    img_blurred = lens.render(img, depth=-1000)

Lens File Formats
-----------------

JSON Format
^^^^^^^^^^^

DeepLens native format with full parameter support:

.. code-block:: json

    {
        "foclen": 50.0,
        "fnum": 1.8,
        "surfaces": [
            {
                "type": "Spheric",
                "r": 46.92,
                "d": 7.0,
                "is_square": false
            }
        ],
        "materials": ["N-BK7", "air"],
        "sensor": {
            "size": [36.0, 24.0],
            "resolution": [4000, 2667]
        }
    }

Zemax Format (.zmx)
^^^^^^^^^^^^^^^^^^^

Load Zemax files directly:

.. code-block:: python

    lens = GeoLens(filename='lens_design.zmx')

Note: Not all Zemax features are supported. Converted to DeepLens format on load.

Lens Properties and Methods
----------------------------

Common Properties
^^^^^^^^^^^^^^^^^

All lens classes share these properties:

.. code-block:: python

    # Focal length [mm]
    print(lens.foclen)
    
    # F-number
    print(lens.fnum)
    
    # Entrance pupil diameter [mm]
    print(lens.enpd)
    
    # Field of view [degrees]
    print(lens.hfov)
    
    # Sensor size [mm]
    print(lens.sensor_size)
    
    # Sensor resolution [pixels]
    print(lens.sensor_res)

Common Methods
^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    
    # PSF calculation
    points = torch.tensor([[0.0, 0.0, -1000.0]])
    psf = lens.psf(points=points, ks=51, spp=2048)
    
    # Image rendering
    img_out = lens.render(img, depth=-1000, method='psf_map')
    
    # Visualization
    lens.plot_setup2D()
    lens.plot_psf(psf)
    
    # Save/load
    lens.write_lens_json('output.json')

Optimization
------------

All lens classes support gradient-based optimization:

.. code-block:: python

    import torch
    
    # Get optimizable parameters with learning rates
    # Learning rates: [d (thickness), c (curvature), k (conic), ai (aspheric)]
    params = lens.get_optimizer_params(
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        decay=0.01
    )
    
    # Use with PyTorch optimizers
    optimizer = torch.optim.Adam(params)
    
    # Optimization loop
    for i in range(1000):
        optimizer.zero_grad()
        
        # Compute loss (e.g., RMS spot size)
        loss = lens.loss_rms(num_grid=9, depth=-10000, num_rays=2048)
        
        loss.backward()
        optimizer.step()

Best Practices
--------------

Performance Tips
^^^^^^^^^^^^^^^^

1. **Use GPU**: Always specify ``device='cuda'`` for significant speedup
2. **Batch Processing**: Process multiple images simultaneously
3. **SPP Selection**: Balance speed vs accuracy (1024-4096 for PSF, 256-512 for rendering)
4. **Mixed Precision**: Use ``torch.cuda.amp`` for faster training

Accuracy Considerations
^^^^^^^^^^^^^^^^^^^^^^^

1. **Ray Tracing**: More rays = better accuracy but slower
2. **Wave Optics**: Use for small F-numbers (< 4) and accurate diffraction
3. **PSFNet**: Fast approximation, may have small errors
4. **Validation**: Always validate against analytical solutions or reference software

Next Steps
----------

* Learn about :doc:`optical_elements` for detailed surface types
* Explore :doc:`sensors` for sensor simulation
* Check :doc:`../examples/automated_lens_design` for optimization examples


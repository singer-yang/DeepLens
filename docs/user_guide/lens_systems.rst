Lens Systems
============

DeepLens provides different simulation models, suited for different types of optical lenses. All of them are differentiable and can be used for lens design and end-to-end optimization.

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
     - High-accuracy simulation for geometric lenses, and differentiable lens design
   * - **DiffracLens**
     - Paraxial wave optics diffractive lens
     - Simple simulation for diffractive optical elements, not accounting for aberrations
   * - **HybridLens**
     - Hybrid refractive-diffractive lens, using ray tracing and wave optics
     - DOEs and metasurfaces with refractive lenses, accounting for aberrations and diffraction
   * - **PSFNetLens**
     - Neural surrogate model
     - Fast PSF prediction compared to ray tracing, image simulation with varying depth and focal plane
   * - **ParaxialLens**
     - Paraxial geometric lens, using Circle of Confusion (CoC) to simulate defocus
     - Quick simulation for defocus, without aberrations

GeoLens - Geometric Ray Tracing
--------------------------------

The ``GeoLens`` class implements a fully differentiable ray tracing engine for refractive/reflective lens systems.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import GeoLens
    
    # Load from file (sensor_res and sensor_size are read from the file)
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Optionally override sensor configuration
    lens.set_sensor(sensor_size=(8.0, 8.0), sensor_res=(2000, 2000))


Supported Surface Types
^^^^^^^^^^^^^^^^^^^^^^^

* **Spheric**: Standard spherical surfaces
* **Aspheric**: Aspheric surfaces with even polynomial terms
* **AsphericNorm**: Aspheric surfaces with normalized even polynomial parameters
* **Plane**: Flat surfaces
* **Aperture**: Aperture stops
* **ThinLens**: Ideal thin lens
* **Cubic**: Cubic phase surfaces
* **Phase**: General phase surfaces simulated with ray tracing

Ray Tracing Methods
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Parallel ray bundle
    ray = lens.sample_parallel(fov_x=0.0, fov_y=0.0)
    ray_out = lens.trace2sensor(ray)
    
    # Sample rays from 3D points
    ray = lens.sample_from_points(
        points=[[0.0, 0.0, -1000.0]],
        num_rays=1024
    )
    ray_out = lens.trace2sensor(ray)

PSF Calculation
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    
    # Single point PSF (center field, at depth -1000mm)
    points = torch.tensor([[0.0, 0.0, -1000.0]])
    psf = lens.psf(points=points, ks=64, spp=4096)
    
    # Off-axis PSF (normalized coordinates)
    points = torch.tensor([[0.5, 0.3, -1000.0]])  # x, y normalized [-1, 1]
    psf = lens.psf(points=points, ks=64, spp=4096)
    
    # RGB PSF
    psf_rgb = lens.psf_rgb(points=points, ks=64, spp=4096)

Image Rendering
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from torchvision.utils import save_image
    
    # Load image as tensor (must match sensor resolution)
    img = torch.rand(1, 3, 2000, 2000).cuda()
    
    # Render through lens using PSF map convolution
    img_rendered = lens.render(
        img,
        depth=-1000,
        method='psf_map',
        psf_grid=(10, 10),
        psf_ks=64
    )
    
    # Or use ray tracing (more accurate, slower and larger memory footprint)
    img_rendered = lens.render(
        img,
        depth=-1000,
        method='ray_tracing',
        spp=32
    )
    
    save_image(img_rendered, 'output.png')

Features
^^^^^^^^

* Fully differentiable ray tracing for lens design optimization
* Support for various refractive and reflective surface types
* Accurate geometric aberration simulation

DiffractiveLens - Paraxial Wave Optics
--------------------------------------

``DiffractiveLens`` implements paraxial wave optics for diffractive optical lenses.

.. code-block:: python

    from deeplens.diffraclens import DiffractiveLens
    
    # Load from file (sensor_res and sensor_size are read from the file)
    lens = DiffractiveLens(
        filename='./datasets/lenses/doe/doe_example.json',
        device='cuda'
    )

Supported Diffractive Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fresnel**: Fresnel zone plates
* **Binary2**: Binary diffractive surfaces
* **Pixel2D**: Pixelated metasurfaces
* **Zernike**: Zernike polynomial surfaces

HybridLens - Refractive-Diffractive Lens System
-----------------------------------------------

``HybridLens`` combines ray tracing and wave optics for accurate simulation of hybrid refractive-diffractive lens systems. However, currently it only supports diffractive surfaces behind a refractive lens.

.. code-block:: python

    import torch
    from deeplens.hybridlens import HybridLens
    
    # Load from file (sensor_res and sensor_size are read from the file)
    lens = HybridLens(
        filename='./datasets/lenses/hybridlens/a489_doe.json',
        device='cuda',
        dtype=torch.float64
    )

    # Calculate PSF
    points = torch.tensor([0.0, 0.0, -10000.0])
    psf = lens.psf(points=points, ks=64, spp=10000000)
    
    # Render image through hybrid lens
    img_rendered = lens.render(img, depth=-1000)

Features
^^^^^^^^

* Accurate optical aberration and diffraction simulation
* Support for DOEs and metasurfaces with refractive lenses

PSFNetLens - Neural Surrogate
------------------------------

``PSFNetLens`` uses neural networks to predict PSFs, enabling fast PSF calculation and image simulation.

.. code-block:: python

    from deeplens import PSFNetLens
    
    # Initialize PSFNetLens with lens file (sensor_res is read from lens file)
    lens = PSFNetLens(
        lens_path='./datasets/lenses/camera/ef50mm_f1.8.json',
        in_chan=3,
        psf_chan=3,
        model_name='mlpconv',
        kernel_size=64
    )
    
    # Load pre-trained network weights
    lens.load_net('./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth')
    
    # Fast image rendering
    img_rendered = lens.render(img, depth=-1000)

Advantages
^^^^^^^^^^

* Faster and more memory efficient than ray tracing and ray-wave model
* Accurate PSF prediction after training
* Differentiable for end-to-end optimization
* Compact model size (can be less than 10MB)
* Supports variant spatial position and focal plane for PSF prediction

Training PSFNet
^^^^^^^^^^^^^^^

To train your own PSFNet model:

.. code-block:: bash

    python 3_psf_net.py

See :doc:`../tutorials` for detailed training instructions.

ParaxialLens - Quick Prototyping
---------------------------------

``ParaxialLens`` implements a paraxial (thin lens) model for rapid simulation for defocus. It uses Circle of Confusion (CoC) to simulate defocus, without aberrations.

.. code-block:: python

    from deeplens.paraxiallens import ParaxialLens
    
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

DeepLens native JSON format (as used by ``GeoLens``):

.. code-block:: json

    {
        "info": "Example lens",
        "foclen": 50.0,
        "fnum": 1.8,
        "r_sensor": 21.6,
        "d_sensor": 72.8,
        "sensor_res": [4000, 2667],
        "surfaces": [
            {
                "type": "Spheric",
                "r": 15.5,
                "c": 0.03,
                "d": 0.0,
                "mat1": "air",
                "mat2": "N-BK7",
                "d_next": 4.5
            },
            {
                "type": "Aperture",
                "r": 9.6,
                "d": 20.3,
                "mat1": "air",
                "mat2": "air",
                "d_next": 5.0
            }
        ]
    }

Zemax Format (.zmx)
^^^^^^^^^^^^^^^^^^^

Load Zemax files directly (currently only supports GeoLens):

.. code-block:: python

    lens = GeoLens(filename='lens_design.zmx')

Note: Not all Zemax features are supported. Converted to DeepLens format on load.

Material Library
-----------------

DeepLens includes extensive material databases for optical glass and plastics:

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
    psf = lens.psf(points=points, ks=64, spp=2048)
    
    # Image rendering
    img_out = lens.render(img, depth=-1000, method='psf_map')
    
    # Visualization
    # Draw an RGB PSF map (available for all lens types)
    lens.draw_psf_map(grid=(7, 7), ks=64, depth=-1000, save_name='psf_map.png')
    # For GeoLens only: draw 2D layout with ray paths
    # lens.draw_layout(filename='layout.png', depth=-1000)
    
    # Save
    lens.write_lens_json('output.json')

Optimization
------------

All lens classes support gradient-based optimization:

.. code-block:: python

    import torch
    
    # Get optimizable parameters with learning rates
    # Learning rates: [d (thickness), c (curvature), k (conic), ai (aspheric)]
    optimizer = lens.get_optimizer(
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        decay=0.01
    )

    
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
4. **Precision**: Use ``dtype=torch.float64`` for phase-critical simulation (e.g., HybridLens)

Accuracy Considerations
^^^^^^^^^^^^^^^^^^^^^^^

1. **GeoLens**: High-accuracy geometric ray tracing; differentiable design; aligns with commercial ray tracers
2. **DiffractiveLens**: Paraxial wave-optics diffraction without geometric aberrations
3. **ParaxialLens**: Fast defocus-only simulation (CoC), no aberrations
4. **HybridLens**: Hybrid refractive-diffractive simulation with aberrations and diffraction
5. **PSFNetLens**: Fast neural approximation; usually accurate enough for image simulation after training
6. **Validation**: Always validate against analytical solutions or reference software

Next Steps
----------

* Learn about :doc:`optical_elements` for detailed surface types
* Explore :doc:`sensors` for sensor simulation
* Check :doc:`../examples/automated_lens_design` for optimization examples


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
    material = Material('N-BK7', wave_range=[450, 650])
    
    # Get refractive index
    n = material.n(wavelength=550)  # nm

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
    ray = lens.sample_point_source(depth=1000, M=512)
    ray_out = lens.trace(ray)
    
    # Parallel ray bundle
    ray = lens.sample_parallel_2D(R=5.0, M=256)
    ray_out = lens.trace(ray)
    
    # Field-dependent sampling
    ray = lens.sample_from_points(
        depth=1000,
        M=256,
        spp=1024,
        field=[0.0, 0.7]  # [x, y] normalized field
    )
    ray_out = lens.trace(ray)

PSF Calculation
^^^^^^^^^^^^^^^

.. code-block:: python

    # Geometric PSF (ray-based)
    psf_ray = lens.psf(depth=1000, spp=4096, method='ray')
    
    # Wave optics PSF (more accurate)
    psf_wave = lens.psf(depth=1000, spp=2048, method='wave')
    
    # Coherent PSF
    psf_coherent = lens.psf(depth=1000, spp=1024, method='coherent')

Image Rendering
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from torchvision.utils import save_image
    
    # Load image as tensor
    img = torch.rand(1, 3, 512, 512).cuda()
    
    # Render through lens
    img_rendered = lens.render(
        img,
        depth=1000,
        spp=512,
        method='fft'  # or 'conv'
    )
    
    save_image(img_rendered, 'output.png')

DiffracLens - Wave Optics
--------------------------

``DiffracLens`` implements wave optics for diffractive optical elements.

.. code-block:: python

    from deeplens import DiffracLens
    
    lens = DiffracLens(
        filename='./datasets/lenses/doe/doe_example.json',
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
        device='cuda',
        wave_method='asm'  # Angular spectrum method
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
    
    # Load pre-trained model
    lens = PSFNetLens(
        ckpt_path='./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth',
        device='cuda'
    )
    
    # Fast PSF prediction
    psf = lens.psf(
        depth=1000,
        field=[0.0, 0.5],  # Field position
        wvln=0.589  # Wavelength in micrometers
    )
    
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
        foclen=50.0,  # Focal length in mm
        fnum=2.0,     # F-number
        sensor_res=(512, 512),
        device='cuda'
    )
    
    # Defocus blur simulation
    img_blurred = lens.render(img, depth=1000, focus_depth=2000)

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

    # PSF calculation
    psf = lens.psf(depth, spp, method)
    
    # Image rendering
    img_out = lens.render(img, depth, spp)
    
    # Visualization
    lens.plot_setup2D()
    lens.plot_psf(psf)
    
    # Save/load
    lens.write_lens_json('output.json')

Optimization
------------

All lens classes support gradient-based optimization:

.. code-block:: python

    # Enable optimization parameters
    lens.set_optimizer_params({
        'radius': True,
        'thickness': True,
        'ai': True
    })
    
    # Get optimizable parameters
    params = lens.parameters()
    
    # Use with PyTorch optimizers
    optimizer = torch.optim.Adam(params, lr=0.01)
    
    # Optimization loop
    for i in range(1000):
        optimizer.zero_grad()
        loss = compute_loss(lens)
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


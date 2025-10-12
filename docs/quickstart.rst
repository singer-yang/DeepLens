Quick Start Guide
=================

This guide will help you get started with DeepLens in just a few minutes.

Hello DeepLens
--------------

Let's create a simple lens system and perform ray tracing.

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens import GeoLens
    
    # Create a lens from a JSON file
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_res=(256, 256),
        device='cuda'
    )
    
    # Print lens information
    print(f"Focal length: {lens.foclen:.2f} mm")
    print(f"F-number: {lens.fnum:.2f}")
    print(f"Field of view: {lens.hfov:.2f} degrees")

Ray Tracing
^^^^^^^^^^^

Perform ray tracing through the lens system:

.. code-block:: python

    from deeplens.optics import Ray
    
    # Create a ray bundle
    ray = lens.sample_parallel_2D(R=5.0, M=256)
    
    # Trace rays through the lens
    ray_out = lens.trace(ray)
    
    # Visualize the results
    lens.plot_setup2D(M=5, plot_rays=True)

Point Spread Function (PSF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate and visualize the PSF:

.. code-block:: python

    # Calculate PSF
    psf = lens.psf()
    
    # Visualize PSF
    lens.plot_psf(psf)

Image Rendering
^^^^^^^^^^^^^^^

Render an image through the lens system:

.. code-block:: python

    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load an image
    img = Image.open('./datasets/bird.png')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    # Render through lens
    img_rendered = lens.render(img_tensor, depth=1e4)
    
    # Save the result
    from torchvision.utils import save_image
    save_image(img_rendered, 'rendered_image.png')

Working with Different Lens Types
----------------------------------

GeoLens (Geometric Lens)
^^^^^^^^^^^^^^^^^^^^^^^^^

For traditional refractive lens systems using ray tracing:

.. code-block:: python

    from deeplens import GeoLens
    
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )

PSFNetLens (Neural Surrogate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fast PSF simulation using neural networks:

.. code-block:: python

    from deeplens import PSFNetLens
    
    lens = PSFNetLens(
        ckpt_path='./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth',
        device='cuda'
    )
    
    # Fast PSF calculation
    psf = lens.psf(depth=1000, field=[0.0, 0.0])

HybridLens (Refractive-Diffractive)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For hybrid refractive-diffractive lens systems:

.. code-block:: python

    from deeplens import HybridLens
    
    lens = HybridLens(
        filename='./datasets/lenses/hybridlens/hybrid_example.json',
        device='cuda'
    )

Camera System
-------------

Combine a lens with a sensor:

.. code-block:: python

    from deeplens import Camera
    from deeplens.sensor import RGBSensor
    
    # Create camera
    camera = Camera(
        lens=lens,
        sensor=RGBSensor(),
        device='cuda'
    )
    
    # Capture image
    image = camera.capture(scene, depth)

Next Steps
----------

* Explore the :doc:`tutorials` for more detailed examples
* Check out the :doc:`api/lens` for detailed API documentation
* See :doc:`examples/automated_lens_design` for advanced applications

Repository Structure
--------------------

The DeepLens repository is organized as follows::

    DeepLens/
    ├── deeplens/              # Main package
    │   ├── optics/            # Optical simulation modules
    │   ├── sensor/            # Sensor simulation modules
    │   ├── network/           # Neural network architectures
    │   ├── geolens.py         # Geometric lens class
    │   ├── diffraclens.py     # Diffractive lens class
    │   ├── hybridlens.py      # Hybrid lens class
    │   └── psfnetlens.py      # Neural surrogate lens
    ├── 0_hello_deeplens.py    # Basic tutorial
    ├── 1_end2end_lens_design.py    # End-to-end design example
    ├── 2_autolens_rms.py      # Automated lens design
    └── configs/               # Configuration files

Running Example Scripts
-----------------------

DeepLens comes with several example scripts:

.. code-block:: bash

    # Basic tutorial
    python 0_hello_deeplens.py
    
    # End-to-end lens design
    python 1_end2end_lens_design.py
    
    # Automated lens design
    python 2_autolens_rms.py
    
    # PSF network training
    python 3_psf_net.py
    
    # Task-specific lens design
    python 4_tasklens_img_classi.py

Each script includes detailed comments and configuration options.


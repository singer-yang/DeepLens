Quick Start Guide
=================

This guide will help you get started with DeepLens in just a few minutes.

Hello DeepLens
--------------

Let's create a simple geometric lens system and perform ray tracing.

Create a lens (GeoLens for example)
^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens import GeoLens
    
    # Create a lens from a JSON file
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_res=(256, 256),
        device='cuda'
    )
    
    # Draw lens layout
    lens.analysis()

Point Spread Function (PSF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate and visualize the PSF of the lens:

.. code-block:: python

    # Calculate PSF
    psf = lens.psf(points=torch.tensor([[0.0, 0.0, -10000.0]]))
    
    # Visualize PSF
    lens.plot_psf(psf.squeeze(0).cpu().numpy())

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
    img_rendered = lens.render(img_tensor, depth=-10000.0, method="ray_tracing")
    
    # Save the result
    from torchvision.utils import save_image
    save_image(img_rendered, 'rendered_image.png')

Working with Different Lens Types
----------------------------------

DeepLens supports various types of lens models to suit different simulation needs and computational requirements:

* **GeoLens**: Traditional refractive lens systems using ray tracing
* **HybridLens**: Hybrid refractive-diffractive lens systems using ray tracing and wave optics
* **PSFNetLens**: Fast neural network-based PSF surrogate models
* **ParaxialLens**: Simple paraxial/ABCD matrix model for defocus simulation
* **DiffractiveLens**: Pure diffractive optical elements using wave propagation

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
    
    # Create PSFNetLens with the original lens file
    lens = PSFNetLens(
        lens_path='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_res=(3000, 3000)
    )
    
    # Load pretrained network weights
    lens.load_net('./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth')
    
    # Fast PSF calculation
    psf_rgb = lens.psf_rgb(points=torch.tensor([[0.0, 0.0, -10000.0]]))

HybridLens (Refractive-Diffractive)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For hybrid refractive-diffractive lens systems:

.. code-block:: python

    from deeplens import HybridLens
    
    lens = HybridLens(
        filename='./datasets/lenses/hybridlens/a489_doe.json',
        device='cuda'
    )

ParaxialLens (Paraxial Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple paraxial lens model with Circle of Confusion (CoC):

.. code-block:: python

    from deeplens import ParaxialLens
    
    lens = ParaxialLens(
        foclen=50.0,
        fnum=1.8,
        sensor_size=(36.0, 24.0),
        sensor_res=(2000, 2000),
        device='cuda'
    )
    
    # Refocus to a specific distance
    lens.refocus(foc_dist=-1000.0)

DiffractiveLens (Diffractive Optics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pure diffractive optical elements using wave propagation:

.. code-block:: python

    from deeplens import DiffractiveLens
    
    # Load from file
    lens = DiffractiveLens(
        filename='./datasets/lenses/diffraclens/doelens.json',
        sensor_size=(8.0, 8.0),
        sensor_res=(2000, 2000),
        device='cuda'
    )
    
    # Or create a simple example
    lens = DiffractiveLens.load_example1()

Camera System
-------------

Combine a lens with an image sensor:

.. code-block:: python

    from deeplens import Camera
    from deeplens.sensor import RGBSensor
    
    # Create camera
    camera = Camera(
        lens=lens,
        sensor=RGBSensor(),
        device='cuda'
    )
    
    # Simulate an image
    image = camera.render(data_dict, render_mode="psf_patch", output_type="rggbif")

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


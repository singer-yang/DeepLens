Quick Start Guide
=================

This guide will help you get started with DeepLens in just a few minutes.

Hello DeepLens
--------------

Let's create a simple geometric lens system and perform ray tracing.

Create a lens (GeoLens for example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens import GeoLens
    
    # Create a lens from a JSON file
    lens = GeoLens(
        filename='./datasets/lenses/camera/ef50mm_f1.8.json',
        device='cuda'
    )
    
    # Optionally change sensor resolution (keeps sensor radius fixed)
    lens.set_sensor_res(sensor_res=(256, 256))
    
    # Draw lens layout, spot diagram, and MTF
    lens.analysis(save_name='./lens_analysis')

Point Spread Function (PSF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate and visualize the PSF of the lens using different models ("geometric", "coherent", "huygens"):

.. code-block:: python

    # Calculate PSF for a point source
    # points: [x, y, z] where x,y are normalized [-1, 1], z is depth in mm (negative)
    point = torch.tensor([0.0, 0.0, -10000.0])  # On-axis, 10m away
    
    # Geometric PSF (incoherent ray tracing)
    psf_geo = lens.psf(points=point, ks=128, spp=16384, model="geometric")
    
    # Coherent PSF (Ray-Wave model)
    psf_coh = lens.psf(points=point, ks=128, model="coherent")
    
    # Huygens PSF (Spherical wave integration)
    psf_huy = lens.psf(points=point, ks=128, model="huygens")
    
    # Visualize PSF across the field
    lens.draw_psf_radial(save_name='./psf_radial.png', depth=-10000.0)

Image Rendering
^^^^^^^^^^^^^^^

Render an image through the lens system:

.. code-block:: python

    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    # Load an image
    img = Image.open('./datasets/bird.png')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    # Resize to match sensor resolution (required for ray_tracing method)
    img_tensor = transforms.functional.resize(img_tensor, lens.sensor_res[::-1])
    
    # Render through lens
    # Methods: 'ray_tracing' (accurate), 'psf_map' (efficient), 'psf_patch' (single PSF)
    img_rendered = lens.render(img_tensor, depth=-10000.0, method='ray_tracing', spp=32)
    
    # Save the result
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
        lens_path='./datasets/lenses/camera/ef50mm_f1.8.json'
    )
    
    # Load pretrained network weights
    lens.load_net('./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth')
    
    # Fast PSF calculation
    psf_rgb = lens.psf_rgb(points=torch.tensor([[0.0, 0.0, -10000.0]]))

HybridLens (Refractive-Diffractive)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For hybrid refractive-diffractive lens systems:

.. code-block:: python

    import torch
    from deeplens.hybridlens import HybridLens
    
    # Set double precision for accurate wave optics
    torch.set_default_dtype(torch.float64)
    
    lens = HybridLens(
        filename='./datasets/lenses/hybrid/hybridlens_example.json',
        device='cuda'
    )
    lens.double()  # Ensure double precision for coherent ray tracing

ParaxialLens (Paraxial Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simple paraxial lens model with Circle of Confusion (CoC):

.. code-block:: python

    from deeplens.paraxiallens import ParaxialLens
    
    lens = ParaxialLens(
        foclen=50.0,           # Focal length in mm
        fnum=1.8,              # F-number
        sensor_size=(36.0, 24.0),  # Sensor size in mm
        sensor_res=(2000, 2000),   # Sensor resolution in pixels
        device='cuda'
    )
    
    # Refocus to a specific distance (negative = object in front)
    lens.refocus(foc_dist=-1000.0)

DiffractiveLens (Diffractive Optics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pure diffractive optical elements using wave propagation:

.. code-block:: python

    from deeplens.diffraclens import DiffractiveLens
    
    # Load from file
    lens = DiffractiveLens(
        filename='./datasets/lenses/diffractive/doelens.json',
        device='cuda'
    )
    
    # Or create a simple example with Fresnel DOE
    lens = DiffractiveLens.load_example1()
    lens.to('cuda')

Camera System
-------------

Combine a lens with an image sensor:

.. code-block:: python

    from deeplens import Camera
    
    # Create camera with lens and sensor configuration files
    camera = Camera(
        lens_file='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_file='./datasets/sensors/imx586.json',
        lens_type='geolens',  # 'geolens' or 'hybridlens'
        device='cuda'
    )
    
    # Prepare input data dictionary
    data_dict = {
        'img': img_tensor,           # sRGB image [B, 3, H, W], range [0, 1]
        'iso': torch.tensor([100]),  # ISO value
        'field_center': torch.tensor([[0.0, 0.0]])  # Field center [-1, 1]
    }
    
    # Simulate camera-captured image with aberrations and noise
    data_lq, data_gt = camera.render(data_dict, render_mode='psf_patch', output_type='rggbif')

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


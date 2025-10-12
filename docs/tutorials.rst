Tutorials
=========

This section provides comprehensive tutorials for using DeepLens.

Tutorial Series
---------------

The DeepLens repository includes several tutorial scripts that cover different aspects of the library:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Script
     - Description
   * - ``0_hello_deeplens.py``
     - Introduction to DeepLens basics: loading lenses, ray tracing, PSF calculation, and image rendering
   * - ``1_end2end_lens_design.py``
     - End-to-end lens optimization with vision tasks
   * - ``2_autolens_rms.py``
     - Automated lens design using curriculum learning
   * - ``3_psf_net.py``
     - Training neural surrogate models for fast PSF prediction
   * - ``4_tasklens_img_classi.py``
     - Task-specific lens design for image classification
   * - ``5_pupil_field.py``
     - Pupil and field analysis
   * - ``6_hybridlens_design.py``
     - Hybrid refractive-diffractive lens design
   * - ``7_comp_photography.py``
     - Computational photography applications

Tutorial 1: Loading and Analyzing Lenses
-----------------------------------------

Loading Lens Files
^^^^^^^^^^^^^^^^^^

DeepLens supports multiple lens file formats:

.. code-block:: python

    from deeplens import GeoLens
    
    # Load from JSON format
    lens = GeoLens(filename='./datasets/lenses/camera/ef50mm_f1.8.json')
    
    # Load from Zemax format (.zmx)
    lens = GeoLens(filename='./datasets/lenses/zemax_double_gaussian.zmx')

Lens Properties
^^^^^^^^^^^^^^^

Access key lens properties:

.. code-block:: python

    print(f"Focal length: {lens.foclen:.2f} mm")
    print(f"F-number: {lens.fnum:.2f}")
    print(f"Entrance pupil diameter: {lens.enpd:.2f} mm")
    print(f"Field of view: {lens.hfov:.2f} degrees")
    print(f"Number of surfaces: {len(lens.surfaces)}")

Visualization
^^^^^^^^^^^^^

Visualize the lens system:

.. code-block:: python

    # 2D cross-section with ray tracing
    lens.plot_setup2D(M=10, plot_rays=True)
    
    # 3D visualization
    lens.plot_setup3D()

Tutorial 2: Ray Tracing
-----------------------

Creating Ray Bundles
^^^^^^^^^^^^^^^^^^^^

Different ways to create rays:

.. code-block:: python

    # Parallel rays (for on-axis analysis)
    ray = lens.sample_parallel_2D(R=5.0, M=256)
    
    # Point source rays (for PSF calculation)
    ray = lens.sample_point_source(
        depth=1000.0,  # Distance from lens
        M=256,         # Number of rays
        R=lens.entrance_pupilr
    )
    
    # Field-dependent rays
    ray = lens.sample_from_points(
        depth=1000.0,
        M=256,
        spp=100,
        field=[0.0, 0.7]  # Normalized field coordinates
    )

Ray Tracing
^^^^^^^^^^^

Trace rays through the lens:

.. code-block:: python

    # Forward ray tracing
    ray_out = lens.trace(ray)
    
    # Check which rays reached the sensor
    valid_rays = ray_out.ra > 0
    print(f"Valid rays: {valid_rays.sum().item()} / {ray.o.shape[0]}")

Tutorial 3: Point Spread Function (PSF)
----------------------------------------

Basic PSF Calculation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Calculate PSF at infinity focus
    psf = lens.psf(
        depth=1e4,
        spp=2048,
        method='wave'  # or 'ray' for geometric PSF
    )
    
    # Visualize
    lens.plot_psf(psf)

PSF Across the Field
^^^^^^^^^^^^^^^^^^^^

Calculate PSF map across different field positions:

.. code-block:: python

    # PSF map calculation
    psf_map = lens.psf_map(
        depth=1000.0,
        spp=1024
    )
    
    # Visualize PSF map
    lens.plot_psf_map(psf_map)

Depth-Varying PSF
^^^^^^^^^^^^^^^^^

Analyze defocus effects:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    depths = [500, 1000, 2000, 5000, 10000]
    
    fig, axes = plt.subplots(1, len(depths), figsize=(15, 3))
    for i, depth in enumerate(depths):
        psf = lens.psf(depth=depth, spp=1024)
        axes[i].imshow(psf[0, 0].cpu())
        axes[i].set_title(f'{depth} mm')
        axes[i].axis('off')
    plt.show()

Tutorial 4: Image Rendering
----------------------------

Basic Image Rendering
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    # Load image
    img = Image.open('./datasets/bird.png')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    # Render through lens
    img_rendered = lens.render(
        img_tensor,
        depth=1000.0,
        spp=256
    )
    
    # Save result
    save_image(img_rendered, 'output.png')

Depth-Aware Rendering
^^^^^^^^^^^^^^^^^^^^^

Render scenes with depth variation:

.. code-block:: python

    # Load RGB and depth
    img_rgb = Image.open('./datasets/edof/rgb.png')
    img_depth = Image.open('./datasets/edof/depth.png')
    
    rgb_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).cuda()
    depth_map = transforms.ToTensor()(img_depth).unsqueeze(0).cuda()
    
    # Scale depth appropriately (e.g., to millimeters)
    depth_map = depth_map * 5000 + 500  # 500mm to 5500mm
    
    # Render with depth
    img_rendered = lens.render_depth(rgb_tensor, depth_map, spp=256)
    
    save_image(img_rendered, 'depth_rendered.png')

Tutorial 5: Lens Optimization
------------------------------

Basic Optimization Setup
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.optim as optim
    
    # Set up optimization
    lens.set_optimizer_params({
        'radius': True,      # Optimize surface radii
        'thickness': True,   # Optimize thicknesses
        'material': False,   # Keep materials fixed
        'conic': True,       # Optimize conic constants
        'ai': True          # Optimize aspheric coefficients
    })
    
    # Create optimizer
    optimizer = optim.Adam(lens.parameters(), lr=0.01)

Optimization Loop
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import SpotLoss
    
    loss_fn = SpotLoss()
    
    for iteration in range(1000):
        optimizer.zero_grad()
        
        # Sample rays
        ray = lens.sample_point_source(depth=1e4, M=256)
        
        # Trace rays
        ray_out = lens.trace(ray)
        
        # Calculate loss
        loss = loss_fn(ray_out)
        
        # Add constraints
        loss += lens.loss_constraint()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.6f}")

Tutorial 6: Using Neural Surrogates
------------------------------------

PSFNetLens
^^^^^^^^^^

Fast PSF prediction using neural networks:

.. code-block:: python

    from deeplens import PSFNetLens
    
    # Load pre-trained model
    lens = PSFNetLens(
        ckpt_path='./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth'
    )
    
    # Fast PSF calculation
    psf = lens.psf(
        depth=1000.0,
        field=[0.0, 0.5],
        wvln=0.589
    )
    
    # Much faster than geometric ray tracing!
    img_rendered = lens.render(img_tensor, depth=1000.0)

Training a Surrogate Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

See ``3_psf_net.py`` for a complete example of training your own PSF network:

.. code-block:: bash

    python 3_psf_net.py

Tutorial 7: Camera Systems
---------------------------

Creating a Camera
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import Camera, GeoLens
    from deeplens.sensor import RGBSensor
    
    # Create lens
    lens = GeoLens(filename='./datasets/lenses/camera/ef50mm_f1.8.json')
    
    # Create sensor
    sensor = RGBSensor(
        resolution=(1920, 1080),
        pixel_size=4.0e-3  # 4 micrometers
    )
    
    # Create camera
    camera = Camera(lens=lens, sensor=sensor)

Image Signal Processing
^^^^^^^^^^^^^^^^^^^^^^^

Apply ISP pipeline:

.. code-block:: python

    from deeplens.sensor import ISP
    
    # Create ISP
    isp = ISP(
        demosaic='bilinear',
        white_balance=True,
        gamma_correction=True
    )
    
    # Process raw sensor data
    raw_data = camera.capture_raw(scene, depth)
    processed_img = isp(raw_data)

Configuration Files
-------------------

DeepLens supports YAML configuration files for reproducible experiments:

.. code-block:: yaml

    # configs/1_end2end_lens_design.yml
    lens:
      filename: './datasets/lenses/camera/ef50mm_f1.8.json'
      sensor_res: [256, 256]
    
    optimization:
      learning_rate: 0.01
      iterations: 1000
      loss_type: 'rms_spot'
    
    constraints:
      min_thickness: 0.5
      max_thickness: 20.0

Load configuration:

.. code-block:: python

    import yaml
    
    with open('configs/1_end2end_lens_design.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    lens = GeoLens(**config['lens'])

Next Steps
----------

* Check the :doc:`examples/automated_lens_design` for advanced applications
* Explore the :doc:`api/lens` for detailed API documentation
* Join our community on `Slack <https://join.slack.com/t/deeplens/shared_invite/zt-2wz3x2n3b-plRqN26eDhO2IY4r_gmjOw>`_


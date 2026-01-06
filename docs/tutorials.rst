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

    import math
    print(f"Focal length: {lens.foclen:.2f} mm")
    print(f"F-number: {lens.fnum:.2f}")
    # Compute entrance pupil diameter from pupil parameters
    _, entr_r = lens.get_entrance_pupil()
    print(f"Entrance pupil diameter: {2 * entr_r:.2f} mm")
    print(f"Horizontal FoV: {math.degrees(lens.hfov):.1f} deg")
    print(f"Number of surfaces: {len(lens.surfaces)}")

Visualization
^^^^^^^^^^^^^

Visualize the lens system:

.. code-block:: python

    # 2D layout with ray tracing
    lens.draw_layout(filename="layout.png", depth=float("inf"))

    # 3D visualization (saved to directory)
    lens.draw_lens_3d(save_dir="./vis3d")

Tutorial 2: Ray Tracing
-----------------------

Creating Ray Bundles
^^^^^^^^^^^^^^^^^^^^

Different ways to create rays:

.. code-block:: python

    # 2D parallel rays (for layout/on-axis analysis)
    ray = lens.sample_parallel_2D(fov=0.0, num_rays=11, plane="sagittal")

    # 3D parallel rays (for geometric analysis)
    ray = lens.sample_parallel(fov_x=0.0, fov_y=0.0, num_rays=1024)

    # Point source rays (object at 1 m; object-space depths are negative)
    ray = lens.sample_point_source(fov_x=0.0, fov_y=0.0, depth=-1000.0, num_rays=2048)

    # Rays from absolute 3D points in object space
    ray = lens.sample_from_points(points=[[0.0, 0.0, -10000.0]], num_rays=2048)

Ray Tracing
^^^^^^^^^^^

Trace rays through the lens:

.. code-block:: python

    # Trace (returns output rays and optional intersection records)
    ray_out, _ = lens.trace(ray)

    # Check which rays reached the sensor
    num_valid = int(ray_out.valid.sum().item())
    num_total = ray_out.valid.numel()
    print(f"Valid rays: {num_valid} / {num_total}")

Tutorial 3: Point Spread Function (PSF)
----------------------------------------

Basic PSF Calculation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt

    # Single-point PSF at 10 m, centered field (normalized x=y=0)
    psf = lens.psf(points=torch.tensor([0.0, 0.0, -10000.0]), ks=64, spp=2048)

    # Visualize
    plt.imshow(psf.cpu(), cmap="inferno")
    plt.axis("off")
    plt.show()

PSF Across the Field
^^^^^^^^^^^^^^^^^^^^

Calculate PSF map across different field positions:

.. code-block:: python

    # Compute and save PSF map across field
    psf_map = lens.psf_map(depth=-10000.0, grid=(7, 7), ks=64, spp=1024)
    lens.draw_psf_map(grid=(7, 7), ks=64, depth=-10000.0, save_name="psf_map.png")

Depth-Varying PSF
^^^^^^^^^^^^^^^^^

Analyze defocus effects:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    depths = [-5000, -10000, -20000]
    
    fig, axes = plt.subplots(1, len(depths), figsize=(15, 3))
    for i, depth in enumerate(depths):
        psf = lens.psf(points=torch.tensor([0.0, 0.0, depth]), ks=64, spp=1024)
        axes[i].imshow(psf.cpu(), cmap="inferno")
        axes[i].set_title(f'{abs(depth)} mm')
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
    
    # Match sensor resolution to image for full-frame rendering
    lens.set_sensor_res(sensor_res=(img_tensor.shape[-1], img_tensor.shape[-2]))

    # Render through lens (ray tracing)
    img_rendered = lens.render(img_tensor, depth=-10000.0, method="ray_tracing", spp=256)
    
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
    
    # Scale depth to millimeters and use negative sign for object space
    depth_map = - (depth_map * 5000 + 500)  # 500mm to 5500mm -> -[500, 5500] mm

    # Render with depth using PSF interpolation
    img_rendered = lens.render_rgbd(rgb_tensor, depth_map, method="psf_map", psf_grid=(10, 10), psf_ks=64)
    
    save_image(img_rendered, 'depth_rendered.png')

Tutorial 5: Lens Optimization
------------------------------

Basic Optimization Setup
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Get optimizer for lens parameters
    optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-2, 1e-4], decay=0.01)

Optimization Loop
^^^^^^^^^^^^^^^^^

.. code-block:: python

    for iteration in range(1000):
        optimizer.zero_grad()

        # RMS spot error across field (geometric objective)
        loss = lens.loss_rms(num_grid=9, depth=-10000.0, num_rays=2048)

        # Regularization for physical feasibility (spacing, angles, thickness)
        loss_reg, _ = lens.loss_reg()
        loss = loss + 0.05 * loss_reg

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

    import torch
    from deeplens import PSFNetLens

    # Initialize and load pretrained PSF network
    lens = PSFNetLens(
        lens_path='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_res=(3000, 3000)
    )
    lens.load_net('./ckpts/psfnet/PSFNet_ef50mm_f1.8_ps10um.pth')

    # Fast PSF calculation (RGB PSF)
    psf_rgb = lens.psf_rgb(points=torch.tensor([[0.0, 0.0, -10000.0]]), ks=64)

    # Rendering via PSF map using the surrogate
    img_rendered = lens.render(img_tensor, depth=-10000.0, method='psf_map')

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

    from deeplens import Camera

    camera = Camera(
        lens_file='./datasets/lenses/camera/ef50mm_f1.8.json',
        sensor_file='./datasets/sensors/canon_r6.json',
        device='cuda'
    )

Image Signal Processing
^^^^^^^^^^^^^^^^^^^^^^^

Apply ISP pipeline:

.. code-block:: python

    from deeplens.sensor.isp import InvertibleISP

    # Create ISP
    isp = InvertibleISP(bit=10, black_level=64, bayer_pattern='rggb')

    # Process RAW Bayer data to RGB
    # raw_bayer = ...  # shape (B, 1, H, W)
    # rgb = isp(raw_bayer)

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


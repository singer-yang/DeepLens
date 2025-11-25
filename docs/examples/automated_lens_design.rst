Automated Lens Design
=====================

This example demonstrates automated lens design using curriculum learning and differentiable optimization.

Overview
--------

Automated lens design involves optimizing lens parameters to achieve desired optical performance without manual intervention. DeepLens enables this through:

* **Curriculum learning**: Gradually increasing optimization difficulty
* **Differentiable ray tracing**: End-to-end gradient backpropagation
* **GPU acceleration**: Fast iteration for complex designs
* **Physical constraints**: Ensuring manufacturability

Related Paper
^^^^^^^^^^^^^

This example is based on:

    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, 
    "Curriculum learning for ab initio deep learned refractive optics," 
    *Nature Communications* 2024.

Example: Double Gauss Lens Design
----------------------------------

Let's design a 50mm f/1.8 double Gauss lens from scratch.

Step 1: Initial Setup
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.optim as optim
    from deeplens import GeoLens
    from deeplens.optics.geometric_surface import Spheric, Aspheric, Aperture
    from deeplens.optics import Material
    
    # Create lens from an existing design file
    # The easiest way is to start from a pre-defined lens:
    lens = GeoLens(
        filename='./datasets/lenses/camera/double_gauss.json',
        device='cuda'
    )
    
    # Or create a new lens by loading surfaces from a JSON file
    # Manually constructing surfaces requires careful parameter setup
    # See the JSON format in datasets/lenses/camera/ for examples

Step 2: Configure Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Initialize constraints for lens design
    # Constraints include minimum/maximum thicknesses, air gaps, surface shapes, etc.
    # Default values are set based on lens type (cellphone vs camera)
    lens.init_constraints()
    
    # Get optimizer with learning rates for different parameters:
    # [d, c, k, a] = [thickness, curvature, conic, aspheric coefficients]
    # For camera lenses, typical values are [1e-3, 1e-4, 0, 0]
    # For cellphone lenses, use [1e-4, 1e-4, 1e-1, 1e-4]
    optimizer = lens.get_optimizer(
        lrs=[1e-3, 1e-4, 0, 0],  # Learning rates for [d, c, k, a]
        decay=0.01,               # Decay for higher-order coefficients
        optim_mat=False           # Whether to optimize materials
    )

Step 3: Curriculum Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.basics import DEPTH, WAVE_RGB, SPP_PSF
    
    # Use the built-in optimize() method for curriculum learning
    # This handles ray sampling, loss computation, and constraints automatically
    lens.optimize(
        lrs=[1e-3, 1e-4, 1e-1, 1e-4],  # Learning rates for [d, c, k, a]
        decay=0.01,
        iterations=5000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        result_dir='./results/double_gauss_design'
    )
    
    # Or implement custom training loop:
    for epoch in range(1000):
        optimizer.zero_grad()
        
        # Compute RMS loss across field points
        # loss_rms() samples rays on a grid and computes spot RMS error
        loss_rms = lens.loss_rms(
            num_grid=(7, 7),      # Grid of field points
            depth=DEPTH,          # Object depth (-20000.0 mm default)
            num_rays=SPP_PSF,     # Rays per field point (16384 default)
            sample_more_off_axis=True
        )
        loss_rms_avg = loss_rms.mean()
        
        # Add regularization losses for constraints
        loss_reg, loss_dict = lens.loss_reg(
            w_focus=10.0,      # Weight for focus loss
            w_ray_angle=2.0,   # Weight for ray angle constraints
            w_intersec=1.0,    # Weight for intersection avoidance
            w_gap=0.1,         # Weight for gap constraints
            w_surf=1.0         # Weight for surface shape constraints
        )
        
        # Total loss
        total_loss = loss_rms_avg + 0.05 * loss_reg
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, RMS Loss: {loss_rms_avg.item():.6f}")
            
            # Correct lens shape and visualize
            lens.correct_shape()
            lens.analysis(save_name=f'./lens_epoch{epoch}')

Step 4: Add Aspheric Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # For aspheric optimization, increase learning rates for conic and aspheric coefficients
    # For cellphone lenses: [1e-4, 1e-4, 1e-1, 1e-4]
    optimizer = lens.get_optimizer(
        lrs=[1e-4, 1e-4, 1e-1, 1e-4],  # [d, c, k, a] with higher k and a
        decay=0.01,
        optim_mat=False
    )
    
    # Continue optimization with aspherics
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Compute RMS loss
        loss_rms = lens.loss_rms(
            num_grid=(9, 9),
            depth=-10000.0,
            sample_more_off_axis=True
        )
        loss_rms_avg = loss_rms.mean()
        
        # Regularization
        loss_reg, loss_dict = lens.loss_reg()
        total_loss = loss_rms_avg + 0.05 * loss_reg
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
            
            # Analyze spot size
            with torch.no_grad():
                rms_results = lens.analysis_spot(num_field=5, depth=-10000.0)
                print(f"RMS spot results: {rms_results}")

Step 5: Evaluation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Final evaluation
    print("\\n=== Final Evaluation ===")
    
    # Full analysis: draws layout, spot diagram, MTF, and computes RMS
    lens.analysis(
        save_name='./optimized_lens',
        depth=-10000.0,
        render=True,           # Render a test image
        render_unwarp=False,
        lens_title='Optimized Double Gauss',
        show=False
    )
    
    # RMS spot analysis
    rms_results = lens.analysis_spot(num_field=5, depth=-10000.0)
    print(f"Spot analysis: {rms_results}")
    
    # Draw specific visualizations
    lens.draw_layout(filename='./lens_layout.png', depth=-10000.0, show=False)
    lens.draw_spot_radial(save_name='./lens_spot.png', depth=-10000.0, show=False)
    lens.draw_mtf(save_name='./lens_mtf.png', depth_list=[-10000.0], show=False)
    
    # PSF visualization
    point = torch.tensor([0.0, 0.0, -10000.0])  # Normalized [x, y, z]
    psf = lens.psf(points=point, ks=128, spp=16384)
    lens.draw_psf_radial(save_name='./lens_psf.png', depth=-10000.0, show=False)
    
    # Save design
    lens.write_lens_json('./optimized_lens.json')

Running the Example
-------------------

The complete script is available as ``2_autolens_rms.py``:

.. code-block:: bash

    python 2_autolens_rms.py

Or with custom configuration:

.. code-block:: bash

    python 2_autolens_rms.py --config configs/2_auto_lens_design.yml

Configuration File
------------------

Example ``configs/2_auto_lens_design.yml``:

.. code-block:: yaml

    lens:
      sensor_res: [512, 512]
      sensor_size: [8.0, 8.0]
      target_foclen: 50.0
      target_fnum: 1.8
    
    optimization:
      learning_rate: 0.01
      total_epochs: 1000
      curriculum:
        - stage: 1
          epochs: 200
          depths: [10000]
          fields: [[0, 0]]
        - stage: 2
          epochs: 200
          depths: [10000]
          fields: [[0, 0], [0, 0.3]]
        - stage: 3
          epochs: 300
          depths: [10000]
          fields: [[0, 0], [0, 0.5], [0, 0.7]]
        - stage: 4
          epochs: 300
          depths: [500, 1000, 2000, 5000]
          fields: [[0, 0], [0, 0.5], [0, 0.7]]
    
    constraints:
      min_thickness: 0.5
      max_thickness: 20.0
      min_radius: 10.0
      max_radius: 1000.0

Advanced Techniques
-------------------

Multi-Wavelength Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.basics import WAVE_RGB  # [0.656, 0.588, 0.486] um
    
    # The loss_rms() function already handles multi-wavelength optimization
    # It computes RMS for R, G, B wavelengths and averages them
    loss_rms = lens.loss_rms(
        num_grid=(7, 7),
        depth=-10000.0,
        num_rays=2048,
        sample_more_off_axis=True
    )
    
    # For custom wavelength handling:
    from deeplens.basics import SPP_PSF
    
    wavelengths = WAVE_RGB  # [0.65627250, 0.58756180, 0.48613270]
    total_loss = 0.0
    
    for wvln in wavelengths:
        # Sample rays at specific wavelength
        ray = lens.sample_grid_rays(
            depth=-10000.0,
            num_grid=(7, 7),
            num_rays=SPP_PSF,
            wvln=wvln,
            uniform_fov=True
        )
        
        # Trace rays to sensor
        ray = lens.trace2sensor(ray)
        
        # Compute RMS error
        rms = ray.rms_error()
        total_loss += rms.mean()

Aberration-Specific Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # DeepLens uses RMS spot size as the primary optimization target,
    # which implicitly minimizes all aberrations that affect spot size.
    
    # For aberration analysis (not optimization loss), use:
    # - lens.draw_spot_radial() for spot diagrams showing aberrations
    # - lens.draw_mtf() for MTF analysis
    # - lens.draw_distortion_radial() for distortion analysis
    
    # Custom loss with weighted field points
    # Off-axis fields have more aberrations, so weight them higher
    loss_rms = lens.loss_rms(
        num_grid=(9, 9),
        depth=-10000.0,
        sample_more_off_axis=True  # Samples more rays at off-axis fields
    )
    
    # The loss_reg() function includes constraints on:
    # - Surface shape (prevents extreme sag/gradient)
    # - Ray angles (controls chief ray angle)
    # - Thickness and air gaps
    loss_reg, loss_dict = lens.loss_reg()

Adaptive Learning Rate
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    for epoch in range(1000):
        loss = train_one_epoch()
        scheduler.step(loss)

Tips and Best Practices
------------------------

1. **Start Simple**: Begin with spherical surfaces, add aspherics later
2. **Curriculum Learning**: Gradually increase optimization difficulty
3. **Physical Constraints**: Always enforce manufacturability constraints
4. **Multi-Field**: Optimize across multiple field points for good off-axis performance
5. **Multi-Wavelength**: Include multiple wavelengths for chromatic correction
6. **Learning Rate**: Start with higher LR (0.01-0.05), reduce as optimization progresses
7. **Monitoring**: Regularly visualize spot diagrams and ray paths
8. **Validation**: Compare with commercial software (Zemax, CodeV)

Expected Results
----------------

After optimization, you should achieve:

* RMS spot size < 10 μm across field
* MTF > 0.3 @ 50 lp/mm
* Distortion < 2%
* Physically realizable geometry

Troubleshooting
---------------

**Loss Not Decreasing**

* Reduce learning rate
* Check constraint violations
* Increase SPP (samples per point)
* Start from better initial design

**Unphysical Designs**

* Strengthen constraints
* Add edge thickness penalties
* Limit optimization variables

**Slow Convergence**

* Use curriculum learning
* Increase batch size (more field points)
* Enable GPU acceleration

See Also
--------

* :doc:`../tutorials` - Tutorial on lens optimization
* :doc:`../api/lens` - GeoLens API reference
* Paper: `Nature Communications 2024 <https://www.nature.com/articles/s41467-024-50835-7>`_


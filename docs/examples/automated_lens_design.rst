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
    from deeplens.optics import Spheric, Aspheric, Aperture
    from deeplens.optics import Material
    
    # Create lens with initial surfaces
    lens = GeoLens(
        sensor_res=(512, 512),
        sensor_size=(8.0, 8.0),
        device='cuda'
    )
    
    # Add surfaces (rough double Gauss layout)
    surfaces = [
        Spheric(r=50.0, d=5.0),    # Front element
        Spheric(r=-50.0, d=2.0),
        Spheric(r=30.0, d=5.0),
        Spheric(r=-30.0, d=15.0),
        Aperture(r=10.0, d=15.0),  # Aperture stop
        Spheric(r=30.0, d=5.0),
        Spheric(r=-30.0, d=2.0),
        Spheric(r=50.0, d=5.0),
        Spheric(r=-50.0, d=40.0),  # Back element
    ]
    
    # Materials
    materials = [
        Material('N-BK7'), Material('air'),
        Material('N-SF11'), Material('air'),
        Material('air'),  # Aperture
        Material('N-SF11'), Material('air'),
        Material('N-BK7'), Material('air'),
    ]
    
    lens.surfaces = surfaces
    lens.materials = materials
    lens.post_computation()

Step 2: Configure Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Enable optimization for specific parameters
    lens.set_optimizer_params({
        'radius': True,       # Optimize radii
        'thickness': True,    # Optimize thicknesses
        'conic': False,       # Keep spherical for now
        'ai': False,          # No aspherics yet
    })
    
    # Set physical constraints
    lens.init_constraints(
        min_thickness=0.5,      # Minimum edge thickness
        max_thickness=20.0,     # Maximum center thickness
        min_radius=10.0,        # Minimum radius
        max_radius=1000.0       # Maximum radius
    )
    
    # Create optimizer
    optimizer = optim.Adam(lens.parameters(), lr=0.01)

Step 3: Curriculum Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import SpotLoss
    
    loss_fn = SpotLoss()
    
    # Curriculum stages
    curricula = [
        # Stage 1: On-axis only
        {'depths': [1e4], 'fields': [[0, 0]], 'epochs': 200},
        # Stage 2: Add near-axis field
        {'depths': [1e4], 'fields': [[0, 0], [0, 0.3]], 'epochs': 200},
        # Stage 3: Full field
        {'depths': [1e4], 'fields': [[0, 0], [0, 0.5], [0, 0.7]], 'epochs': 300},
        # Stage 4: Multiple depths
        {'depths': [500, 1000, 2000, 5000], 
         'fields': [[0, 0], [0, 0.5], [0, 0.7]], 'epochs': 300},
    ]
    
    for stage_idx, stage in enumerate(curricula):
        print(f"\\n=== Stage {stage_idx + 1} ===")
        
        for epoch in range(stage['epochs']):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Optimize over all depths and fields in this stage
            for depth in stage['depths']:
                for field in stage['fields']:
                    # Sample rays
                    ray = lens.sample_from_points(
                        depth=depth,
                        M=32,
                        spp=100,
                        field=field
                    )
                    
                    # Trace rays
                    ray_out = lens.trace(ray)
                    
                    # Calculate loss
                    loss = loss_fn(ray_out)
                    total_loss += loss
            
            # Add constraints
            total_loss += lens.loss_constraint()
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        # Visualize after each stage
        lens.plot_setup2D(M=5, plot_rays=True)
        lens.analysis_rms_spot()

Step 4: Add Aspheric Surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Enable aspheric optimization
    lens.set_optimizer_params({
        'radius': True,
        'thickness': True,
        'conic': True,        # Enable conic constants
        'ai': True,           # Enable aspheric coefficients
    })
    
    # Continue optimization with aspherics
    optimizer = optim.Adam(lens.parameters(), lr=0.005)
    
    for epoch in range(500):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Full optimization
        for depth in [500, 1000, 2000, 5000]:
            for field in [[0, 0], [0, 0.5], [0, 0.7], [0.7, 0]]:
                ray = lens.sample_from_points(
                    depth=depth, M=32, spp=100, field=field
                )
                ray_out = lens.trace(ray)
                total_loss += loss_fn(ray_out)
        
        total_loss += lens.loss_constraint()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
            rms = lens.analysis_rms_spot()
            print(f"RMS spot size: {rms.mean():.3f} μm")

Step 5: Evaluation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Final evaluation
    print("\\n=== Final Evaluation ===")
    
    # RMS spot diagram
    lens.analysis_rms_spot()
    
    # Distortion
    lens.analysis_distortion()
    
    # MTF
    mtf = lens.analysis_mtf(frequency=50)
    print(f"Average MTF @ 50 lp/mm: {mtf.mean():.3f}")
    
    # PSF visualization
    psf = lens.psf(depth=1000, spp=4096)
    lens.plot_psf(psf)
    
    # Save design
    lens.write_lens_json('optimized_lens.json')

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

    wavelengths = [0.486, 0.550, 0.656]  # Blue, green, red
    
    for wvln in wavelengths:
        ray = lens.sample_from_points(
            depth=depth, M=32, spp=100, 
            field=field, wavelength=wvln
        )
        ray_out = lens.trace(ray)
        total_loss += loss_fn(ray_out)

Aberration-Specific Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import SphericalAberrationLoss, ComaLoss
    
    sa_loss = SphericalAberrationLoss()
    coma_loss = ComaLoss()
    
    # Combined loss
    loss = spot_loss(ray_out) + 0.5 * sa_loss(ray_out) + 0.5 * coma_loss(ray_out)

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


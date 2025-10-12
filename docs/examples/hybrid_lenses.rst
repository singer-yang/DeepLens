Hybrid Lens Design
==================

Design and simulation of hybrid refractive-diffractive optical systems.

Overview
--------

Hybrid lenses combine:

* **Refractive elements**: Traditional glass lenses
* **Diffractive elements**: DOEs, metasurfaces, diffractive optics

Benefits:

* **Chromatic correction**: Diffractive elements have opposite dispersion
* **Compact form factor**: Thin diffractive elements replace thick glass
* **Novel functionalities**: Wavefront shaping, multi-focal, etc.

Related Paper
^^^^^^^^^^^^^

    Congli Wang, Ni Chen, and Wolfgang Heidrich, 
    "dflens: A Differentiable Pipeline for Hybrid Refractive-Diffractive Lens Design," 
    *SIGGRAPH Asia 2024*.

Basic Hybrid Lens
-----------------

Creating a Hybrid System
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens import HybridLens
    from deeplens.optics import Spheric, Aperture
    from deeplens.optics.diffractive_surface import Fresnel, Pixel2D
    from deeplens.optics import Material
    
    # Create hybrid lens
    lens = HybridLens(
        sensor_res=(512, 512),
        sensor_size=(8.0, 8.0),
        wave_method='asm',  # Angular spectrum method
        device='cuda'
    )
    
    # Add refractive elements
    lens.surfaces.append(Spheric(r=50.0, d=5.0))
    lens.materials.append(Material('N-BK7'))
    
    lens.surfaces.append(Spheric(r=-50.0, d=10.0))
    lens.materials.append(Material('air'))
    
    # Add diffractive element (Fresnel zone plate)
    lens.surfaces.append(Fresnel(
        foclen=50.0,
        d=0.001,  # Very thin
        zone_num=100,
        wavelength=0.550
    ))
    lens.materials.append(Material('air'))
    
    # Aperture
    lens.surfaces.append(Aperture(r=10.0, d=30.0))
    lens.materials.append(Material('air'))
    
    lens.post_computation()

Ray-Wave Hybrid Tracing
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create rays
    ray = lens.sample_point_source(depth=1000, M=128)
    
    # Trace through hybrid system
    # Refractive surfaces use ray tracing
    # Diffractive surfaces use wave propagation
    field_out = lens.trace_hybrid(ray)
    
    # Calculate PSF from field
    psf = torch.abs(field_out) ** 2

Metasurface Design
------------------

Pixelated Metasurface
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.diffractive_surface import Pixel2D
    
    # Create height map for metasurface
    # Each pixel can have different height
    height_map = torch.zeros(1024, 1024)
    
    # Design focusing metasurface
    for i in range(1024):
        for j in range(1024):
            x = (i - 512) * 0.5  # μm
            y = (j - 512) * 0.5  # μm
            r2 = x**2 + y**2
            
            # Focusing phase profile
            # φ = -k * r^2 / (2f)
            phase = -2 * torch.pi * r2 / (2 * 50000 * 0.550)
            
            # Convert phase to height
            # h = φ * λ / (2π * (n-1))
            height = phase * 0.550 / (2 * torch.pi * (1.5 - 1.0))
            height_map[i, j] = height % 0.550  # Wrap to [0, λ]
    
    # Create metasurface
    metasurface = Pixel2D(
        height_map=height_map,
        pixel_size=0.5,  # 500nm pixels
        d=0.001,
        n_material=1.5,
        wavelength=0.550
    )
    
    lens.surfaces.append(metasurface)

Zernike-Based DOE
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.diffractive_surface import Zernike
    
    # Design DOE using Zernike polynomials
    # Coefficients for common aberrations
    coefficients = [
        0,      # Z0: Piston
        0, 0,   # Z1, Z2: Tilt
        -1.0,   # Z3: Defocus
        0, 0,   # Z4, Z5: Astigmatism
        0, 0,   # Z6, Z7: Coma
        0,      # Z8: Trefoil
        0.5,    # Z9: Spherical aberration
    ]
    
    doe = Zernike(
        coefficients=coefficients,
        d=0.001,
        aperture_radius=10.0,
        wavelength=0.550
    )
    
    lens.surfaces.append(doe)

Achromatic Hybrid Lens
----------------------

Combining Refractive and Diffractive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Traditional achromat uses crown + flint glass
    # Hybrid achromat uses glass + DOE
    
    # Positive refractive element
    lens.surfaces.append(Spheric(r=40.0, d=6.0))
    lens.materials.append(Material('N-BK7'))  # Crown glass
    
    lens.surfaces.append(Spheric(r=-40.0, d=0.5))
    lens.materials.append(Material('air'))
    
    # Diffractive element for chromatic correction
    # DOE has opposite dispersion to glass
    lens.surfaces.append(Fresnel(
        foclen=-200.0,  # Negative power
        d=0.001,
        zone_num=80,
        wavelength=0.550
    ))
    lens.materials.append(Material('air'))

Multi-Wavelength Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.optim as optim
    from deeplens.optics import SpotLoss
    
    # Enable optimization
    lens.set_optimizer_params({
        'radius': True,
        'thickness': True,
        'doe_phase': True  # Optimize DOE pattern
    })
    
    optimizer = optim.Adam(lens.parameters(), lr=1e-3)
    loss_fn = SpotLoss()
    
    wavelengths = [0.486, 0.550, 0.656]  # Blue, green, red
    
    for epoch in range(1000):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Optimize for all wavelengths
        for wvln in wavelengths:
            ray = lens.sample_point_source(
                depth=1000,
                M=64,
                wavelength=wvln
            )
            field_out = lens.trace_hybrid(ray)
            loss = loss_fn(field_to_ray(field_out))
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

Multi-Focal Lens
----------------

Design lens with multiple focal points:

.. code-block:: python

    from deeplens.optics.diffractive_surface import Binary2
    
    # Create binary phase pattern for two focal lengths
    # f1 = 50mm, f2 = 100mm
    
    H, W = 512, 512
    phase_pattern = torch.zeros(H, W, dtype=torch.bool)
    
    for i in range(H):
        for j in range(W):
            x = (i - H/2) * 0.01  # mm
            y = (j - W/2) * 0.01
            r2 = x**2 + y**2
            
            # Combined phase for two focal lengths
            phase1 = -torch.pi * r2 / (0.550e-3 * 50.0)
            phase2 = -torch.pi * r2 / (0.550e-3 * 100.0)
            
            # Binary encoding (simplified)
            total_phase = phase1 + phase2
            phase_pattern[i, j] = (total_phase % (2*torch.pi)) > torch.pi
    
    # Add to lens
    multi_focal_doe = Binary2(
        phase_pattern=phase_pattern,
        d=0.001,
        wavelength=0.550
    )
    
    lens.surfaces.append(multi_focal_doe)

Extended Depth of Field
-----------------------

Using Cubic Phase Plate
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics import Cubic
    
    # Add cubic phase element at aperture
    cubic = Cubic(
        r=float('inf'),  # Flat base
        d=0.001,
        alpha=20.0,  # Cubic coefficient
        device='cuda'
    )
    
    lens.surfaces.insert(aperture_idx, cubic)
    
    # Test at multiple depths
    depths = [500, 1000, 2000, 5000]
    psfs = []
    
    for depth in depths:
        psf = lens.psf(depth=depth, spp=2048)
        psfs.append(psf)
    
    # Visualize
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(depths), figsize=(16, 4))
    for i, (psf, depth) in enumerate(zip(psfs, depths)):
        axes[i].imshow(psf[0, 0].cpu())
        axes[i].set_title(f'{depth} mm')
        axes[i].axis('off')
    plt.show()

Optimization for EDoF
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Optimize for consistent PSF across depth range
    depths = torch.linspace(500, 5000, 10)
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = 0.0
        
        # Calculate PSFs at all depths
        psfs = []
        for depth in depths:
            psf = lens.psf(depth=depth.item(), spp=1024)
            psfs.append(psf)
        
        # Loss: Encourage similar PSFs
        psf_stack = torch.stack(psfs, dim=0)
        psf_mean = psf_stack.mean(dim=0)
        
        # Variance loss
        loss = ((psf_stack - psf_mean) ** 2).mean()
        
        # Also maintain good average PSF
        loss += compute_sharpness_loss(psf_mean)
        
        loss.backward()
        optimizer.step()

Polarization Optics
-------------------

Note: Advanced feature available via collaboration

.. code-block:: python

    from deeplens.optics.geometric_phase import GeometricPhaseElement
    
    # Polarization-dependent focusing
    gp_element = GeometricPhaseElement(
        phase_pattern=phase_map,
        polarization='circular',
        handedness='left'
    )
    
    lens.surfaces.append(gp_element)

Fabrication Considerations
---------------------------

Manufacturable DOE Design
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Constraints for fabrication
    lens.init_doe_constraints(
        min_feature_size=0.5,    # μm
        max_aspect_ratio=5.0,    # Height/width
        quantization_levels=8,    # Multi-level DOE
        min_zone_width=10.0      # μm
    )
    
    # Add to optimization loss
    loss += lens.loss_doe_constraint()

Multi-Level DOE
^^^^^^^^^^^^^^^

.. code-block:: python

    # Quantize continuous phase to discrete levels
    num_levels = 8
    
    def quantize_phase(phase, num_levels):
        """Quantize phase to discrete levels."""
        phase_norm = (phase % (2*torch.pi)) / (2*torch.pi)
        phase_quantized = torch.round(phase_norm * num_levels) / num_levels
        return phase_quantized * 2 * torch.pi
    
    # Apply to DOE
    for surf in lens.surfaces:
        if isinstance(surf, (Fresnel, Pixel2D, Zernike)):
            surf.phase_map = quantize_phase(surf.phase_map, num_levels)

Running Example
---------------

Complete script available as ``6_hybridlens_design.py``:

.. code-block:: bash

    python 6_hybridlens_design.py

Performance Comparison
----------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Metric
     - Pure Refractive
     - Hybrid Design
   * - **Chromatic Aberration**
     - Requires multiple elements
     - Single DOE can correct
   * - **Weight**
     - Heavy (multiple glass elements)
     - Light (thin DOEs)
   * - **Thickness**
     - Thick lens stack
     - Compact design
   * - **Cost**
     - High (precision glass)
     - Lower (planar fabrication)
   * - **Efficiency**
     - High (>95%)
     - Moderate (70-90%)
   * - **Diffraction Orders**
     - N/A
     - Manage higher orders

Advantages and Limitations
---------------------------

**Advantages:**

* Compact and lightweight
* Excellent chromatic correction
* Novel functionalities (multi-focal, EDoF)
* Potential for lower cost

**Limitations:**

* Lower diffraction efficiency
* Wavelength-dependent performance
* More complex fabrication
* Stray light from higher orders

Tips and Best Practices
------------------------

1. **Start with Refractive**: Design refractive system first, add DOE for correction
2. **Wavelength Range**: DOEs are wavelength-sensitive, design for target range
3. **Efficiency**: Balance performance with diffraction efficiency
4. **Sampling**: Use high SPP (>2048) for accurate diffraction simulation
5. **Fabrication**: Consider manufacturing constraints early in design
6. **Validation**: Prototype and test critical designs

See Also
--------

* :doc:`automated_lens_design` - Optimization techniques
* :doc:`../user_guide/optical_elements` - Diffractive surfaces
* :doc:`../tutorials` - Step-by-step guides
* Paper: `SIGGRAPH Asia 2024 <https://arxiv.org/abs/2406.00834>`_


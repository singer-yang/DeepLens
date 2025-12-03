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

    import torch
    from deeplens.hybridlens import HybridLens
    
    # HybridLens consists of a GeoLens (refractive) and a DOE (diffractive)
    # The easiest way is to load from a JSON file that contains both:
    lens = HybridLens(
        filename='./datasets/lenses/hybrid/hybridlens_example.json',
        device='cuda'
    )
    
    # The HybridLens has two main components:
    # - lens.geolens: The refractive lens (GeoLens)
    # - lens.doe: The diffractive optical element (DOE)
    
    # For accurate wave optics simulation, use double precision
    torch.set_default_dtype(torch.float64)
    lens.double()
    
    # Access refractive lens properties
    print(f"Focal length: {lens.foclen}")
    print(f"Sensor size: {lens.geolens.sensor_size}")
    
    # The DOE can be: Binary2, Pixel2D, Fresnel, or Zernike type
    print(f"DOE type: {type(lens.doe)}")

Ray-Wave Hybrid Tracing
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.basics import PSF_KS, DEFAULT_WAVE, SPP_COHERENT
    
    # Ensure double precision for accurate phase computation
    torch.set_default_dtype(torch.float64)
    
    # HybridLens uses a ray-wave model:
    # 1. Coherent ray tracing through refractive elements to DOE plane
    # 2. DOE phase modulation
    # 3. Wave propagation (Angular Spectrum Method) to sensor
    
    # Compute PSF using the psf() method
    point = [0.0, 0.0, -10000.0]  # Point source position [x, y, z]
    psf = lens.psf(
        points=point,        # Point source location
        ks=PSF_KS,           # Kernel size (128 default)
        wvln=DEFAULT_WAVE,   # Wavelength (0.58756180 um)
        spp=SPP_COHERENT     # Rays for coherent tracing (~16.7M)
    )
    
    # PSF is normalized intensity distribution, shape [ks, ks]
    print(f"PSF shape: {psf.shape}")
    print(f"PSF sum: {psf.sum():.4f}")  # Should be ~1.0

Metasurface Design
------------------

Pixelated Metasurface
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.diffractive_surface import Pixel2D
    
    # Pixel2D DOE allows arbitrary phase patterns
    # Parameters are typically loaded from JSON or optimized during training
    
    # Example: create a Pixel2D DOE with focusing phase
    import torch
    import math
    
    res = 512  # DOE resolution
    ps = 0.008  # Pixel size in mm (8 um)
    wvln = 0.550  # Design wavelength in um
    foclen = 50.0  # Focal length in mm
    
    # Create focusing phase profile
    x = torch.linspace(-res//2, res//2-1, res) * ps
    y = torch.linspace(-res//2, res//2-1, res) * ps
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    r2 = xx**2 + yy**2
    
    # Focusing phase: φ = -π * r^2 / (λ * f)
    phase = -math.pi * r2 / (wvln * 1e-3 * foclen)
    phase = phase % (2 * math.pi)  # Wrap to [0, 2π]
    
    # Create DOE (see deeplens/optics/diffractive_surface/ for details)
    # In practice, DOEs are defined in JSON files and loaded with HybridLens

Zernike-Based DOE
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from deeplens.optics.diffractive_surface import Zernike
    
    # Zernike DOE uses Zernike polynomial coefficients to define phase
    # Useful for correcting specific aberrations
    
    # DOE parameters are typically defined in JSON:
    # {
    #     "param_model": "zernike",
    #     "d": 45.0,           # DOE position (mm)
    #     "h": 4.0,            # DOE height (mm)
    #     "w": 4.0,            # DOE width (mm)
    #     "res": [512, 512],   # DOE resolution
    #     "coef": [0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0.5]  # Zernike coefficients
    # }
    
    # In HybridLens, the DOE is accessed via lens.doe
    # Zernike coefficients can be optimized during training

Achromatic Hybrid Lens
----------------------

Combining Refractive and Diffractive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # HybridLens combines a GeoLens (refractive) with a DOE (diffractive)
    # The DOE is placed at the end of the optical system
    
    # Benefits of hybrid design:
    # - DOE has opposite chromatic dispersion to glass
    # - Can correct chromatic aberration with a single thin element
    # - Enables compact achromatic designs
    
    # Load a hybrid lens design
    from deeplens.hybridlens import HybridLens
    
    lens = HybridLens(
        filename='./datasets/lenses/hybrid/achromatic_hybrid.json',
        device='cuda'
    )
    
    # Access components
    geolens = lens.geolens  # Refractive part (GeoLens)
    doe = lens.doe          # Diffractive part (Binary2, Pixel2D, Fresnel, or Zernike)
    
    # The DOE compensates for chromatic aberration from refractive elements

Multi-Wavelength Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.optim as optim
    from deeplens.basics import WAVE_RGB, SPP_COHERENT, PSF_KS
    
    # Set double precision for coherent ray tracing
    torch.set_default_dtype(torch.float64)
    
    # Get optimizer for both GeoLens and DOE
    optimizer = lens.get_optimizer(
        doe_lr=1e-4,                      # DOE learning rate
        lens_lr=[1e-4, 1e-4, 1e-2, 1e-5], # GeoLens [d, c, k, a]
        lr_decay=0.01
    )
    
    wavelengths = WAVE_RGB  # [0.656, 0.588, 0.486] um
    point = [0.0, 0.0, -10000.0]  # On-axis point source
    
    for epoch in range(1000):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Optimize for all wavelengths
        for wvln in wavelengths:
            # Compute PSF using ray-wave model
            psf = lens.psf(
                points=point,
                ks=PSF_KS,
                wvln=wvln,
                spp=SPP_COHERENT
            )
            
            # Loss: minimize PSF spread (maximize peak)
            loss = -psf.max()  # Simple peak sharpening loss
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

    # For extended depth of field (EDoF), use a GeoLens with Cubic surface
    # Cubic phase creates a depth-invariant PSF (wavefront coding)
    
    from deeplens import GeoLens
    from deeplens.basics import DEPTH, PSF_KS, SPP_PSF
    
    # Load lens with cubic phase element
    # Cubic surfaces are defined in JSON with type "Cubic"
    lens = GeoLens(
        filename='./datasets/lenses/camera/edof_cubic.json',
        device='cuda'
    )
    
    # Test PSF at multiple depths
    import torch
    depths = [-500.0, -1000.0, -2000.0, -5000.0]
    psfs = []
    
    for depth in depths:
        point = torch.tensor([0.0, 0.0, depth])
        psf = lens.psf(points=point, ks=PSF_KS, spp=SPP_PSF)
        psfs.append(psf)
    
    # Cubic phase creates similar PSF shapes across depth
    # Post-processing with deconvolution recovers sharp images

Optimization for EDoF
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from deeplens.basics import PSF_KS, SPP_PSF
    
    # Optimize DOE for consistent PSF across depth range
    depths = torch.linspace(-500, -5000, 5)
    
    for epoch in range(500):
        optimizer.zero_grad()
        loss = 0.0
        
        # Calculate PSFs at all depths
        psfs = []
        for depth in depths:
            point = torch.tensor([0.0, 0.0, depth.item()])
            psf = lens.psf(points=point, ks=PSF_KS, spp=SPP_PSF)
            psfs.append(psf)
        
        # Loss: Encourage similar PSFs across depths
        psf_stack = torch.stack(psfs, dim=0)
        psf_mean = psf_stack.mean(dim=0)
        
        # Variance loss (minimize PSF variation)
        variance_loss = ((psf_stack - psf_mean) ** 2).mean()
        
        # Sharpness loss (maximize PSF peak)
        sharpness_loss = -psf_mean.max()
        
        loss = variance_loss + 0.1 * sharpness_loss
        
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

    # For manufacturability, consider:
    # 1. Minimum feature size (typically > 0.5 μm for photolithography)
    # 2. Maximum aspect ratio (height/width < 5 for standard processes)
    # 3. Quantization levels (8 or 16 levels for multi-level DOE)
    
    # Apply phase quantization for multi-level DOE
    import torch
    
    def quantize_phase(phase, num_levels=8):
        """Quantize continuous phase to discrete levels."""
        phase_norm = (phase % (2 * torch.pi)) / (2 * torch.pi)
        phase_quantized = torch.round(phase_norm * num_levels) / num_levels
        return phase_quantized * 2 * torch.pi
    
    # Access DOE phase map
    wvln = 0.550
    phase_map = lens.doe.get_phase_map(wvln)
    
    # Quantize for fabrication
    phase_quantized = quantize_phase(phase_map, num_levels=8)

Multi-Level DOE
^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    
    # Quantize continuous phase to discrete levels for fabrication
    num_levels = 8  # Common: 2 (binary), 4, 8, 16 levels
    
    def quantize_phase(phase, num_levels):
        """Quantize phase to discrete levels."""
        phase_norm = (phase % (2 * torch.pi)) / (2 * torch.pi)
        phase_quantized = torch.round(phase_norm * num_levels) / num_levels
        return phase_quantized * 2 * torch.pi
    
    # Get DOE phase map and quantize
    wvln = 0.550  # Design wavelength
    phase_continuous = lens.doe.get_phase_map(wvln)
    phase_multilevel = quantize_phase(phase_continuous, num_levels)
    
    # Note: For training with quantization, use straight-through estimator
    # to allow gradients to flow through the quantization operation

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


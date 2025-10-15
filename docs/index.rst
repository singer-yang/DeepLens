.. image:: ../assets/logo.png
   :align: center
   :alt: DeepLens Logo

**DeepLens** is a differentiable optical lens simulator developed for:

1. Differentiable optical design
2. End-to-end optics-vision co-design
3. Photorealistic image simulation

DeepLens helps researchers build custom differentiable optical systems and computational imaging pipelines with minimal effort.

.. note::
   This documentation is still under development. Some details may contain mistakes or be incomplete. We appreciate your patience and welcome any feedback or corrections.

Mission
-------

1. Next-generation **optical design software** enhanced by differentiable optimization
2. Next-generation **computational cameras** integrating optical encoding with deep learning decoding

Key Features
------------

* **Differentiable Optics**: Leverages gradient backpropagation and differentiable optimization
* **Automated Lens Design**: Enables automated lens design using curriculum learning and GPU acceleration
* **Hybrid Refractive-Diffractive Optics**: Accurate simulation of hybrid lenses (DOEs, metasurfaces)
* **Accurate Image Simulation**: Photorealistic, spatially-varying image simulations
* **Optics-Vision Co-Design**: End-to-end differentiability from optics to vision algorithms

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/lens_systems
   user_guide/optical_elements
   user_guide/sensors
   user_guide/neural_networks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/lens
   api/geolens
   api/optics
   api/sensor
   api/network
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/automated_lens_design
   examples/end2end_design
   examples/image_simulation
   examples/hybrid_lenses

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   code_of_conduct
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

# GeoLens Package

The `geolens_pkg` is a sub-package of DeepLens that provides a comprehensive suite of tools for the classical evaluation, optimization, and analysis of geometric lens systems (`GeoLens`). The functionalities are designed to be accurate and are aligned with industry-standard optical design software like Zemax.

## Key Features

This package offers a wide range of optical performance evaluation capabilities:

-   **Spot Diagram Analysis:** Generate and visualize spot diagrams at various field angles to assess aberrations.
-   **RMS Spot Error:** Calculate RMS spot size maps across different wavelengths and field points.
-   **Distortion Analysis:** Compute and plot distortion maps and curves to quantify image deformation.
-   **Modulation Transfer Function (MTF):** Evaluate the spatial frequency response of the lens system to determine its resolution and contrast performance.
-   **Vignetting Analysis:** Calculate and visualize the reduction in image brightness at the periphery of the field.
-   **3D Visualization:** Render 3D views of the lens system for better understanding of its physical layout.
-   **Tolerance Analysis:** Tools for assessing the impact of manufacturing and assembly errors on lens performance.
-   **Optimization:** Utilities for optimizing lens designs based on various performance metrics.

## Modules

The package is organized into the following modules:

-   `eval.py`: The core module for classical optical performance evaluation. It includes functions for analyzing spot diagrams, distortion, MTF, and more.
-   `optim.py`: Contains tools and functions for optimizing `GeoLens` systems.
-   `vis.py`: Provides utilities for plotting and visualizing the results of various optical analyses.
-   `view_3d.py`: Includes functionalities for creating and displaying 3D renderings of lens systems.
-   `tolerance.py`: Implements functions for performing tolerance analysis on lens parameters.
-   `io.py`: A set of helper functions for input/output operations, such as loading and saving lens data.
-   `utils.py`: Contains various utility functions used across the package.

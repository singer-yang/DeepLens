Installation
============

Prerequisites
-------------

DeepLens requires:

* Python 3.9 or later
* PyTorch with CUDA support (recommended for GPU acceleration)
* Conda (optional, but recommended for environment management)

Installation Methods
--------------------

Method 1: Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the repository::

    git clone https://github.com/singer-yang/DeepLens
    cd DeepLens

Create a conda environment using the provided environment file::

    conda env create -f environment.yml -n deeplens_env
    conda activate deeplens_env

Method 2: Using pip
^^^^^^^^^^^^^^^^^^^

Clone the repository::

    git clone https://github.com/singer-yang/DeepLens
    cd DeepLens

Create a virtual environment and install dependencies::

    conda create --name deeplens_env python=3.9
    conda activate deeplens_env
    pip install -r requirements.txt

Verify Installation
-------------------

To verify that DeepLens is installed correctly, run the demo script::

    python 0_hello_deeplens.py

If the installation is successful, you should see lens visualization and simulation outputs.

GPU Support
-----------

For optimal performance, DeepLens requires a CUDA-capable GPU. To check if PyTorch can detect your GPU::

    python -c "import torch; print(torch.cuda.is_available())"

If this returns ``False``, you may need to reinstall PyTorch with CUDA support. Visit the `PyTorch installation page <https://pytorch.org/get-started/locally/>`_ for instructions.

Additional Dependencies
-----------------------

Some advanced features may require additional packages:

* **Matplotlib**: For visualization (usually included in requirements.txt)
* **OpenCV**: For image processing operations
* **Pillow**: For image I/O operations

These are typically installed automatically with the standard installation methods.

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Import Errors**

If you encounter import errors, ensure that:

1. The conda/virtual environment is activated
2. All dependencies are installed correctly
3. You're running Python from the DeepLens root directory

**CUDA/GPU Issues**

If you have GPU issues:

1. Check that your NVIDIA drivers are up to date
2. Verify that PyTorch is installed with the correct CUDA version
3. Try running with CPU first to isolate the issue

For more help, join our `Slack workspace <https://join.slack.com/t/deeplens/shared_invite/zt-2wz3x2n3b-plRqN26eDhO2IY4r_gmjOw>`_ or contact the developers.


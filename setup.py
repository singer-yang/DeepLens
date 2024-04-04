from setuptools import setup, find_packages
setup(name='deeplens', version='1.0', packages=find_packages())

from setuptools import setup, find_packages

setup(
    name='deeplens',
    version='1.0',
    author='Xinge Yang',
    author_email='xinge.yang@kaust.edu.sa',  # Your email
    description='DeepLens design.',  # A short description
    long_description=open('README.md').read(),  # A long description from your README file
    long_description_content_type='text/markdown',  # This is important for a markdown readme
    url='https://https://github.com/singer-yang/DeepLens',  # Link to your project's GitHub repo or website
    packages=find_packages(),  # Find all packages (directories with __init__.py)
    install_requires=[
        # List your project's dependencies here as strings, e.g.,
        # 'requests >= 2.19.1',
        'opencv-python',
        'matplotlib',
        'scikit-image',
        'h5py',
        'transformers',
        'lpips',
        'einops',
        'timm',
    ],
    license='Creative Commons Attribution-NonCommercial 4.0 International License',
    classifiers=[
        # Choose your license as you wish
        'License :: Other/Proprietary License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. Otherwise, your package will not
        # appear in the PyPI search results for Python 3.
        # For a list of classifiers, check here: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9',  # Your package's Python version compatibility
    # entry_points={
    #     'console_scripts': [
    #         'your_command = your_package.module:function',
    #     ],
    # },  # If your package has command line scripts
)

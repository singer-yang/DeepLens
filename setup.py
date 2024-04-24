from setuptools import setup, find_packages
setup(name='deeplens', version='1.0', packages=find_packages())
from setuptools import setup, find_packages

setup(
    name='deeplens',
    version='1.0',
    author='Xinge Yang',
    author_email='xinge.yang@kaust.edu.sa',
    description='DeepLens: a differentiable ray tracer for computational lens design.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/singer-yang/DeepLens',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'matplotlib',
        'scikit-image',
        'h5py',
        'transformers',
        'lpips',
        'einops',
        'timm',
        'tqdm'
    ],
    classifiers=[
        'License :: CC-BY-4.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)

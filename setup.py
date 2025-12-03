from setuptools import setup, find_packages

setup(
    name='deeplens',
    version='1.0.0',
    author='Xinge Yang',
    author_email='xinge.yang@kaust.edu.sa',
    description='DeepLens: differentiable optical lens simulator for end-to-end cameras.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/singer-yang/DeepLens',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib',
        'scikit-image',
        'transformers',
        'lpips',
        'einops',
        'timm',
        'tqdm',
        'wandb'
    ],
    classifiers=[
        'License :: CC-BY-4.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.12',
)

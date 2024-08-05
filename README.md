<div style="text-align:center;">
    <img src="imgs/logo.png"/>
</div>

**DeepLens** is an open-source differentiable ray tracer for optical design, optics-network co-design, and optics-aware rendering.

Welcome to use DeepLens to (1) build your own pipeline or (2) compare as the baseline. We can provide code assistance if you plan to use DeepLens in your research, please contact Xinge Yang (xinge.yang@kaust.edu.sa) for more information. Manufacturing service for both refractive and diffractive elements is also avaliable via collaboration!

## News

* DeepLens is finall published on Nature Communications, check it [here](https://www.nature.com/articles/s41467-024-50835-7)!
* We now have a WeChat group. Please contact Xinge Yang (singeryang1999) to join the discussion!

## What is DeepLens

**Deep learning + lens design**

1. More powerful **lens design software** with deep learning.
2. Next-generation **computational cameras** (optical encoding + deep learning decoding).

**Key features**

1. **Open-source** ray tracer with similar accuracy as Zemax.
2. **Optimization** power. Automated lens design from scratch!
3. **Render** camera captured image and start optics-network co-design with 5 lines of code!
4. **Memory-efficient** ray-tracing, millions of rays on a desk-machine.
5. **Neural representation**, represent a camera lens as a network.
6. More features can be obtained via request or collaboration:
   1. Hybrid ray-tracing-wave-propagation imaging model.
   2. Non-sequential ray-tracing.
   3. Polarization ray-tracing.
   4. Multi-GPU parallelization for billion-level ray-tracing.
   5. Illumination and display system optimization.

## Applications

#### 1. Automated lens design

Fully automated lens design from scratch. Try it at [AutoLens](https://github.com/vccimaging/AutoLens)!

[![paper](https://img.shields.io/badge/NatComm-2024-orange)](https://www.nature.com/articles/s41467-024-50835-7) [![quickstart](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/AutoLens)

<div align="center">
    <img src="imgs/autolens1.gif" alt="AutoLens" height="270px"/>
    <img src="imgs/autolens2.gif" alt="AutoLens" height="270px"/>
</div>

#### 2. End-to-End lens design

Lens-network co-design from scratch using final images (or classification/detection/segmentation) as objective.

[![paper](https://img.shields.io/badge/NatComm-2024-orange)](https://www.nature.com/articles/s41467-024-50835-7)

<div align="center">
    <img src="imgs/end2end.gif" alt="End2End" height="150px"/>
</div>

#### 3. Implicit Lens Representation

A surrogate network for fast (aberration + defocus) image simulation.

[![paper](https://img.shields.io/badge/TPAMI-2023-orange)](https://ieeexplore.ieee.org/document/10209238) [![link](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/Aberration-Aware-Depth-from-Focus)

<div align="center">
    <img src="imgs/implicit_net.png" alt="Implicit" height="150px"/>
</div>

#### 4. Hybrid Refractive-Difractive Lens Model

Design hybrid refractive-diffractive lenses with a new ray-wave model.

[![report](https://img.shields.io/badge/Arxiv-2024-orange)](https://arxiv.org/abs/2406.00834)

<div align="center">
    <img src="imgs/hybridlens.png" alt="Implicit" height="200px"/>
</div>

## How to use

Here are two methods to use deeplens in your research:

#### Method 1

Clone this repo and write your code inside it.

```
git clone deeplens
cd deeplens
python 0_hello_deeplens.py
python your_code.py
```

#### Method 2

Clone the repo and install deeplens as a python package.

```
git clone deeplens
pip install -e ./deeplens
```

Then in your code:

```
import deeplens
lens = deeplens.GeoLens(filename='./lenses/cellphone80deg.json')
```

#### Directory

```
deeplens/
│
├── deeplens/
│   ├── optics/ (contain core functions for optical components)
|   ├── network/ (contain network architectures for image reconstruction and implicit representation)
|   ├── geolens (lensgroup using ray tracing)
│   └── diffraclens (lensgroup using wave propagation)
│
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
└── 0_hello_deeplens.py (main scripts)

```

## Reference

#### Citations

DeepLens is first developed by [Dr. Congli Wang](https://congliwang.github.io/) (previously named **dO**), then developed and maintained by [Xinge Yang](https://singer-yang.github.io/).

If you use DeeoLens in your research, please cite the corresponding papers:

- [TCI 2022] dO: A differentiable engine for deep lens design of computational imaging systems. [Paper](https://ieeexplore.ieee.org/document/9919421), [BibTex](./misc/do_bibtex.txt)
- [NatComm 2024] Curriculum Learning for ab initio Deep Learned Refractive Optics. [Paper](https://www.nature.com/articles/s41467-024-50835-7), [BibTex](./misc/deeplens_bibtex.txt)
- [Arxiv 2024] End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model. [Paper](https://arxiv.org/abs/2406.00834), [BibTex](./misc/hybridlens_bibtex.txt)

#### Projects built on top of DeepLens/dO

(If you donot want to list your paper here, we can remove it.)

- [TPAMI 2023] Aberration-Aware Depth-From-Focus. [Paper](https://ieeexplore.ieee.org/document/10209238), [BibTex](./misc/aatdff_bibtex.txt)
- [Arxiv 2024] Centimeter-Scale Achromatic Hybrid Metalens Design: A New Paradigm Based on Differentiable Ray Tracing in the Visible Spectrum. [Paper](https://arxiv.org/abs/2404.03173)

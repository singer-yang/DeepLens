# Contributing Guide

Thank you for your interest in contributing to DeepLens!

All contributors are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## How to Contribute

We welcome contributions in various forms, including but not limited to:
- Reporting bugs
- Submitting pull requests with bug fixes or new features
- Improving documentation
- Adding new examples or tutorials

If you plan to work on a major feature, please open an issue to discuss your ideas with the maintainers first.

## Developer Installation

DeepLens is primarily a Pytorch project. To set up your development environment, please follow the "How to use" section in the [README.md](./README.md) to create a conda environment and install the necessary dependencies.

A quick summary of the steps:
```
# Create and activate a conda environment
conda env create -f environment.yml -n deeplens_env
conda activate deeplens_env
```
or
```
conda create --name deeplens_env python=3.9
conda activate deeplens_env
pip install -r requirements.txt
```

## Contribution Opportunities

A great place to start looking for contribution ideas is the project's issue tracker on GitHub. Look for issues with labels like "good first issue" or "help wanted". You can also check out the [open questions project board](https://github.com/users/singer-yang/projects/2) mentioned in the README.

## Proposing Major Changes

For substantial changes to the codebase, it is a good idea to open an issue to propose your change. This allows for discussion with the maintainers and community before you invest significant time in implementation. This helps ensure your contribution aligns with the project's goals and is more likely to be accepted.

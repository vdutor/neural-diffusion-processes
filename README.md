![Quality checks and Tests](https://github.com/vdutor/neural-diffusion-processes/actions/workflows/run-tests.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2206.03992)

# Neural Diffusion Processes

This repository contains the code for the paper [Neural Diffusion Processes](https://arxiv.org/abs/2206.03992) by Vincent Dutordoir, Alan Saul, Zoubin Ghahramani and Fergus Simpson accepted at ICML 2023.

## Installation

The code is written in Python and uses the jax framework with the haiku deep learning library. The code has been tested on ubuntu linux with Python 3.9 and 3.10.

Before installing the dependencies, we recommend creating a virtual environment. Then the dependencies and the source (`neural_diffusion_processes`) can be installed with the following command using `poetry`:
```
poetry install
```
This command will install the required dependencies to import the package and run the tests. To verify that the installation was successful, run the following command:
```
poetry run task test
```

## Experiments

Instructions to reproduce the paper's experiments can be found in the readme files in each of the experiment directories.

To install the dependencies for the experiments, run the following command:
```
poetry install --with experiments
```
This commands installs a lot of extra dependecies that are required for the experiments, such as [aim](https://aimstack.io/) (for experiment tracking), [ml-tools](https://github.com/vdutor/ml-tools), `tensorflow-cpu` (for data loading), etc.

### GPU support
Note that we do not install `jaxlib` by default, which is required for GPU support. To install `jaxlib` with GPU support for your machine following the official instructions [here](https://github.com/google/jax#installation).

#### To cite this work:
```
@InProceedings{dutordoir2023,
  title = {{N}eural {D}iffusion {P}rocesses},
  author = {Dutordoir, Vincent and Saul, Alan and Ghahramani, Zoubin and Simpson, Fergus},
  url = {https://arxiv.org/abs/2206.03992},
  booktitle = {International Conference of Machine Learning},
  year = {2023},
}
```

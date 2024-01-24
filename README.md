# PyAWD: a Python acoustic wave propagation dataset using PyTorch and Devito
This package offers tools for generating a dataset of simulations of the acoustic wave propagation in the Marmousi velocity field. It uses the <a href="https://www.devitoproject.org">Devito Python Library</a> to solve the acoustic wave PDE from various random initial conditions.

## Installation
The package is accessible via <a href="https://pypi.org">PyPI</a>:

`pip install pyawd`

## Documentation
Basic help is provided for each class and function, and is accessible via the python `help()` function.

## Examples
Mutliple IPython notebooks are presented in the `examples` directory:
- `HeatPropagation.ipynb`: an introduction to PDE solving and simulation using Devito applied on the heat propagation
- `AcousticWaveGeneration.ipynb`: an introduction to PDE solving and simulation using Devito applied on the acoustic wave propagation
- `Marmousi.ipynb`: a visualisation of the Marmousi velocity field used in the simulations
- `GenerateAcousticWaveDataset.ipynb`: an example of dataset generation workflow
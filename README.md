# PyAWD: a Python acoustic wave propagation dataset using PyTorch and Devito
A package for generating a Pytorch dataset containing simulations of the acoustic wave propagation in the Marmousi velocity field. It uses the [Devito Python Library](https://www.devitoproject.org) to solve the acoustic wave PDE from various random initial conditions.

## Marmousi velocity field
![Marmousi velocity field](https://slideplayer.com/slide/15021598/91/images/37/Marmousi+Velocity+Model.jpg)

<img src="https://slideplayer.com/slide/15021598/91/images/37/Marmousi+Velocity+Model.jpg" alt="Marmousi velocity field" width="20%"/>

## Installation
The package is accessible via [PyPI](https://pypi.org):

```
pip install pyawd
```

## Documentation
Basic help is provided for each class and function, and is accessible via the python `help()` function.

## Examples
Mutliple IPython notebooks are presented in the [examples](examples/) directory. If [Jupyter](https://jupyter.org) is installed, those examples can be explored by starting jupyter:

```
jupyter-notebook
```

- `HeatPropagation.ipynb`: an introduction to PDE solving and simulation using Devito applied on the heat propagation
- `AcousticWaveGeneration.ipynb`: an introduction to PDE solving and simulation using Devito applied on the acoustic wave propagation
- `Marmousi.ipynb`: a visualisation of the Marmousi velocity field used in the simulations
- `GenerateAcousticWaveDataset.ipynb`: an example of dataset generation workflow

## To do
- Enhance the velocity model by taking into account $P$ and $S$ waves (see [this article](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FThe-Marmousi-II-model_fig1_269276540&psig=AOvVaw1dSJMMNk22p8xyXNmHMVx8&ust=1706178866091000&source=images&cd=vfe&opi=89978449&ved=0CBUQ3YkBahcKEwjwzdff6fWDAxUAAAAAHQAAAAAQEg))
# PyAWD: a Python acoustic wave propagation dataset using PyTorch and Devito
A package for generating a Pytorch dataset containing simulations of the acoustic wave propagation in the Marmousi velocity field. It uses the [Devito Python Library](https://www.devitoproject.org) to solve the acoustic wave PDE from various random initial conditions.

## Marmousi velocity field
The Marmousi velocity field used in the simulation is a subset of the following:

<img src="https://slideplayer.com/slide/15021598/91/images/37/Marmousi+Velocity+Model.jpg" alt="Marmousi velocity field" width="40%"/>

## Installation
The package (along with the dependencies) is accessible via [PyPI](https://pypi.org/project/PyAWD/):

```bash
pip install pyawd
```

## Getting started

Basic imports:
```python
import PyAWD
from PyAWD.ScalarAcousticWaveDataset import ScalarAcousticWaveDataset
```

Let us generate a Dataset made of 10 simulations. Each simulation is run in a $250\times 250$ matrix. We store the field state every $2$ seconds and we run the simulation for $10$ seconds:

```python
dataset = ScalarAcousticWaveDataset(2, nx=250, dt=2, t=10)
```

Then we plot the first simulation. The &#128960; character shows the interrogator position:

```python
dataset.plot_item(0)
```

Which outputs the following figure:
![Example of simulation output](examples/example.png)

Finally, we can generate a video of this simulation. We will use $200$ frames, which yields a final rate of $20 fps$:

```python
dataset.generate_video(0, "example", 200)
```

This produces the following video (stored in the file `example.mp4`):

![Example of simulation output](examples/dataset_generation.gif)


By default, the point `(0, 0)` contains an interrogator. This means that the continuous measurement on this position (at least with a $\Delta t=dt$) can be obtained by:

```python
dataset.interrogate((0, 0))
```


## Documentation
Basic help is provided for each class and function, and is accessible via the Python `help()` function.

## Examples
Mutliple IPython notebooks are presented in the [examples](examples/) directory. If [Jupyter](https://jupyter.org) is installed, those examples can be explored by starting Jupyter:

```bash
jupyter-notebook
```

- `ScalarAcousticWaveGeneration.ipynb`: an introduction to PDE solving and simulation using Devito applied on the scalar acoustic wave propagation
- `VectorialAcousticWaveGeneration.ipynb`: an introduction to PDE solving and simulation using Devito applied on the vectorial acoustic wave propagation
- `Marmousi.ipynb`: a visualisation of the Marmousi velocity field used in the simulations
- `GenerateAcousticWaveDataset.ipynb`: an example of dataset generation workflow
- `Interrogators.ipynb`: an introduction to the PyAWD Interrogators usage

## Related Works:
- https://essd.copernicus.org/preprints/essd-2023-470/

<img src="https://github.com/pascaltribel/pyawd/raw/aca41d51f3a8aaf0028fdfc1b0637e0e545da432/Logo.png" alt="Logo" width="60%"/>

# PyAWD: a Python acoustic wave propagation dataset using PyTorch and Devito
A package for generating Pytorch datasets containing simulations of the acoustic wave propagation in custom velocity fields. 

## Acoustic Wave Equation
The equation of propagation of an acoustic wave is given by $\frac{d^2u}{dt^2} = c \nabla^2 u + f$, where
- $u(x, y, t)$ is the displacement field and can be either a scalar or a vector field
- $c(x, y, t)$ is the wave  propagation speed
- $\nabla^2$ is the _laplacian operator_
- $f(x, y, t)$ is an external force applied on the system, for which the value can vary through time
PyAWD uses the [Devito Python Library](https://www.devitoproject.org) to solve the acoustic wave PDE from various random initial conditions.

## Installation
The package (along with the dependencies) is accessible via [PyPI](https://pypi.org/project/PyAWD/):

```bash
pip install pyawd
```

## Documentation
The API documentation is available [here](https://pascaltribel.github.io/pyawd/).
Basic help is provided for each class and function and is accessible via the Python `help()` function.
We provide a bunch of Notebooks to start using the tool. They are presented in the [examples](https://github.com/pascaltribel/pyawd/tree/main/examples) directory. Those are readable online:
- [`ScalarAcousticWavePropagation.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/ScalarAcousticWavePropagation.html): an introduction to PDE solving and simulation using Devito applied on the scalar acoustic wave propagation
- [`VectorAcousticWavePropagation.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/VectorAcousticWavePropagation.html): an introduction to PDE solving and simulation using Devito applied on the vector acoustic wave propagation
- [`VectorAcousticWaveDataset.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/VectorAcousticWaveDataset.html): an introduction to the VectorAcousticWaveDataset possibilities
- [`Marmousi.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/Marmousi.html): a visualisation of the Marmousi velocity field used in the simulations
- [`Interrogators.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/Interrogators.html): an introduction to the PyAWD Interrogators usage
- [`GenerateVectorAcousticWaveDataset.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/GenerateVectorAcousticWaveDataset.html): how to generate dataset using `pyawd`
- [`SpatioTemporalVaryingWavePropagationSpeedField.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/SpatioTemporalVaryingWavePropagationSpeedField.html): how to create a spatio-temporal varying propagation field

## Getting started

Basic imports:
```python
from pyawd import *
```

Let us generate a Dataset made of 10 simulations. Each simulation is run in a $250\times 250$ matrix. We store the field state every $2$ seconds and we run the simulation for $10$ seconds:

```python
dataset = VectorAcousticWaveDataset2D(2, nx=128, dt=2, t=10, velocity_model="Marmousi")
dataset.max_velocities[0] = 500
```

Then we plot the first simulation. The &#128960; character shows the interrogator position:

```python
dataset.plot_item(0)
```

Which outputs the following figure:

<img src="https://github.com/pascaltribel/pyawd/raw/main/examples/example.png" alt="Example" width="60%"/>

By default, the point `(0, 0)` contains an interrogator. This means that the continuous measurement on this position (at least with a $\Delta t=ddt$) can be plotted by:

```python
dataset.plot_interrogators_response(0)
```

<img src="https://github.com/pascaltribel/pyawd/raw/main/examples/interrogator_example.png" alt="Example" width="60%"/>

## More advanced usage
Using the `VectorAcousticWaveDataset3D` class allows producing simulations in 3D:


```python
dataset_3d = VectorAcousticWaveDataset3D(1, nx=32, t=10, interrogators=[(0, 10, 15), (0, -10, 15)], velocity_model=300.)
```

For visualisation, the method

```python
dataset_3d.generate_video(0, "VAWD3D", 300)
```

generates the following video:

<video src="https://github.com/pascaltribel/pyawd/raw/main/examples/VAWD3D.mp4" width="1000" controls></video>

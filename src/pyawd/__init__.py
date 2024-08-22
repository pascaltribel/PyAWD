# pyawd
# Tribel Pascal - pascal.tribel@ulb.be
r"""
<img src="https://github.com/pascaltribel/pyawd/raw/main/Logo.png?raw=true" alt="Logo" width="60%"/>

# pyawd - A PyTorch Acoustic Wave Dataset Generator
Pyawd (standing for Pytorch Acoustic Wave Dataset) is a powerful tool for building datasets containing custom simulations of the propagation of Acoustic Wave through a given medium.
It uses the finite differences scheme (implemented in the Devito package) to solve the Acoustic Wave Equation, and offers convenient tools for the customisation of the parameters, the handling of the data, the visualisation of the simulations.

Here is an example of the kind of visualisation PyAWD offers. The two first pannels show the propagation in 3 dimensions, and the three last ones show the measured movement at the three interrogators, placed at the cube top surface.

<video src="https://github.com/pascaltribel/pyawd/raw/main/examples/test3.mp4" width="1000" controls></video>


## Acoustic Wave Equation
The equation of propagation of an acoustic wave is given by 
$$\frac{d^2u}{dt^2} = c \nabla^2 u + f(x, y, t)$$ 
where
- $u(x, y, t)$ is the displacement field, and can be either a scalar or a vector field
- $c(x, y, t)$ is the wave  propagation speed
- $\nabla^2$ is the <i>laplacian operator</i>
- $f(x, y, t)$ is an external force applied on the system


## Installation
The package (along with the dependencies) is accessible via [PyPI](https://pypi.org/project/PyAWD/):

```bash
pip install pyawd
```

## Documentation
The API documentation is available [here](https://pascaltribel.github.io/pyawd/).
Basic help is provided for each class and function, and is accessible via the Python `help()` function.

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

By default, the point `(0, 0)` contains an interrogator. This means that the continuous measurement on this position (at least with a $\Delta t=ddt$) can be plot by:

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


## Examples
Multiple IPython notebooks are presented in the [examples](https://github.com/pascaltribel/pyawd/tree/main/examples) directory. If [Jupyter](https://jupyter.org) is installed, those examples can be explored by starting Jupyter:

```bash
jupyter-notebook
```

However, those are provided in HTML for your convenience. Just click on their name in the following list:

- [`ScalarAcousticWavePropagation.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/ScalarAcousticWavePropagation.html): an introduction to PDE solving and simulation using Devito applied on the scalar acoustic wave propagation
- [`VectorAcousticWavePropagation.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/VectorAcousticWavePropagation.html): an introduction to PDE solving and simulation using Devito applied on the vector acoustic wave propagation
- [`VectorAcousticWaveDataset.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/VectorAcousticWaveDataset.html): an introduction to the VectorAcousticWaveDataset possibilities
- [`Marmousi.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/Marmousi.html): a visualisation of the Marmousi velocity field used in the simulations
- [`Interrogators.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/Interrogators.html): an introduction to the PyAWD Interrogators usage
- [`GenerateVectorAcousticWaveDataset.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/GenerateVectorAcousticWaveDataset.html): how to generate dataset using `pyawd`
- [`SpatioTemporalVaryingWavePropagationSpeedField.ipynb`](https://pascaltribel.github.io/pyawd/pyawd/SpatioTemporalVaryingWavePropagationSpeedField.html): how to create a spatio-temporal varying propagation field
"""
from pyawd.VelocityModel import VelocityModel
from pyawd.VelocityModel2D import VelocityModel2D
from pyawd.VelocityModel3D import VelocityModel3D
from pyawd.Marmousi import Marmousi
from pyawd.AcousticWaveDataset import AcousticWaveDataset
from pyawd.VectorAcousticWaveDataset import VectorAcousticWaveDataset
from pyawd.VectorAcousticWaveDataset2D import VectorAcousticWaveDataset2D
from pyawd.VectorAcousticWaveDataset3D import VectorAcousticWaveDataset3D

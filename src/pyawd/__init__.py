# pyawd
# Tribel Pascal - pascal.tribel@ulb.be
r"""
# pyawd
Pyawd (standing for Pytorch Acoustic Wave Dataset) is a powerful tool for building datasets containing custom simulations of the propagation of Acoustic Wave through a given medium.
It uses the finite differences scheme (implemented in the Devito package) to solve the Acoustic Wave Equation, and offers convenient tools for the customisation of the parameters, the handling of the data, the visualisation of the simulations.

## Acoustic Wave Equation
The equation of propagation of an acoustic wave is given by $\frac{d^2u}{dt^2} = c \nabla^2 u + f(x, y)$, where
- $u(x, y)$ is the displacement field, and can be either a scalar or a vector field
- $c(x, y)$ is the wave  propagation speed
- $\nabla^2$ is the _laplacian operator_
- $f(x, y)$ is an external force applied on the system, for which the value can vary through time
"""
from pyawd.ScalarAcousticWaveDataset import ScalarAcousticWaveDataset
from pyawd.VectorAcousticWaveDataset import VectorAcousticWaveDataset
from pyawd.Marmousi import Marmousi

# pyawd
# Tribel Pascal - pascal.tribel@ulb.be
"""
Pyawd (standing for Pytorch Acoustic Wave Dataset) is a powerful tool for building datasets containing custom simulations of the propagation of Acoustic Wave through a given medium.
It uses the finite differences scheme (implemented in the Devito package) to solve the Acoustic Wave Equation, and offers convenient tools for the customisation of the parameters, the handling of the data, the visualisation of the simulations.
"""
from pyawd.ScalarAcousticWaveDataset import ScalarAcousticWaveDataset
from pyawd.VectorialAcousticWaveDataset import VectorialAcousticWaveDataset
from pyawd.Marmousi import Marmousi

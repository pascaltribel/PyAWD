# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from typing import Tuple, List, Dict, Union

import numpy as np
import devito as dvt

import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.colors

COLORS = matplotlib.colors.TABLEAU_COLORS
dvt.configuration['log-level'] = "WARNING"

class AcousticWaveDataset(torch.utils.data.Dataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field
    """
    size: int
    """The number of samples to generate in the dataset"""
    nx: int
    """The discretisation size of the array (maximum size is currently 955)"""
    sx: float
    """The sub-scaling factor of the array (0.5 means $\\frac{1}{2}$ values are returned)"""
    dim: int
    """The number of dimensions of the simulations (2 or 3)"""
    ddt: float
    """The time step used for the Operator solving iterations"""
    dt: float
    """The time step used for storing the wave propagation step (this should be higher than ddt)"""
    ndt: int
    """The number of steps in the simulation, accessible for the interrogators"""
    t: float
    """The simulations duration"""
    nt: int
    """The number of steps in the simulations, for which the whole simulation is accessible"""
    interrogators: List[Tuple]
    """A list containing the coordinates of each interrogator"""
    interrogators_data: Dict[Tuple, List]
    """The measurements of each interrogator"""
    grid: dvt.Grid
    """The devito Grid on which the equation is solved"""
    velocity_model: dvt.Function
    """The propagation speed of the wave"""
    attenuation_factor: float
    """The attenuation factor in the acoustic wave equation"""
    max_velocities: np.ndarray
    """The maximal velocity in the idx propagation field"""
    epicenters: np.ndarray
    """The epicenter of each simulation"""
    force_delay: np.ndarray
    """The delay of apparition of the external force for each simulation"""
    amplitude_factor: np.ndarray
    """The amplitude factor to multiply the external force with"""
    data: np.ndarray
    """The simulations data"""
    keep_full_data: bool
    """Whether keeping the full simulation or only the interrogable one"""

    def __init__(self, size: int, dx: float = 1000/128., nx: int = 128, sx: float = 1., dim: int = 2, ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, velocity_model: Union[
                 str, float] = 300., attenuation_factor: float = 0.5, keep_full_data: bool = True):
        """
        Args:
            size (int): The number of samples to generate in the dataset
            dx (float): The discretisation rate of the array
            nx (int): The discretisation size of the array
            dim (int): The number of dimensions of the simulations (2 or 3)
            sx (float): The sub-scaling factor of the array (0.5 means 1/2 values are returned)
            ddt (float): The time step used for the Operator solving iterations
            dt (float): The time step used for storing the wave propagation step (this should be higher than ddt)
            t (float): The simulations duration
            velocity_model (str | float): either:
                - A string identifier specifying a velocity framework
                - A float, specifying a constant wave propagation speed
            attenuation_factor (float): The attenuation factor in the acoustic wave equation
            keep_full_data (bool): Whether keeping the full simulation or only the interrogable one
        """
        pass

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        pass

    def solve_pde(self, idx: int):
        """
        Solves the Acoustic Wave Equation for the idx parameters.
        Returns:
            (numpy.ndarray): A numpy array containing the solutions for the `ndt` steps
        """
        pass

    def interrogate(self, idx: int, point: Tuple) -> np.ndarray:
        """
        Args:
            idx (int): The number of the sample to interrogate
            point (Tuple): The interrogator position
        Returns:
            (numpy.ndarray): The amplitude measurements for the interrogator at coordinates `point` for the $idx^{th}$ sample
        """
        if point not in self.interrogators_data:
            print("Error: the interrogated point is not interrogable.")
            print("Available interrogable points:", list(self.interrogators_data.keys()))
        else:
            return self.interrogators_data[point][idx]

    def plot_item(self, idx: int):
        """
        Plots the simulation of the $idx^{th}$ sample
        Args:
            idx (int): The number of the sample to plot
        """
        pass

    def plot_interrogators_response(self, idx: int):
        """
        Plots the measurements taken by the interrogators for the $idx^{th}$ sample.
        Args:
            idx (int): The number of the sample to plot
        """
        pass

    def generate_video(self, idx: int, filename: str, nb_images: int):
        """
        Generates a video representing the simulation of the $idx^{th}$ sample propagation
        Arguments:
            idx (int): the number of the sample to simulate in the video
            filename (str): the name of the video output file (without extension)
                        The video will be stored in a file called `filename`.mp4
            nb_images (int): the number of frames used to generate the video. This should be an entire divider of the number
                         of points computed when applying the solving operator
        """
        pass

    def set_scaling_factor(self, sx: float):
        """
        Fixes a new scaling factor (0.5 means $\\frac{1}{2}$ values are returned). It should be <= 1.
        Args:
            sx (float): the new scaling factor
        """
        if sx <= 1.:
            self.sx = sx
        else:
            print("The scaling factor should be lower or equal to 1.")

    def __len__(self):
        """
        Returns:
            (int): The number of simulations in the dataset
        """
        return self.size

    def __getitem__(self, idx):
        """
        Returns:
            (Tuple): The epicenter, the simulation of the `idx`th sample, the maximal speed of propagation of the
             propagation field, the delay before the external force application and the force amplitude factor
        """
        pass
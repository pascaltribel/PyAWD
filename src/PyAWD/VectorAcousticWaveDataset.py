# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from memory_profiler import profile
from typing import Tuple, List

import numpy as np
import devito as dvt
from matplotlib.colors import TABLEAU_COLORS

from pyawd import AcousticWaveDataset

COLORS = TABLEAU_COLORS
dvt.configuration['log-level'] = "WARNING"


class VectorAcousticWaveDataset(AcousticWaveDataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field
    """

    def __init__(self, size: int, dx: float = 1000/128., nx: int = 128, sx: float = 1., ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, velocity_model: str | float = 300.,
                 attenuation_factor: float = 0.5, keep_full_data: bool = True):
        """
        Args:
            size (int): The number of samples to generate in the dataset
            dx (float): The discretisation rate of the array
            nx (int): The discretisation size of the array
            sx (float): The sub-scaling factor of the array (0.5 means 1/2 values are returned)
            ddt (float): The time step used for the Operator solving iterations
            dt (float): The time step used for storing the wave propagation step (this should be higher than ddt)
            t (float): The simulations duration
            velocity_model (str | float): either:
                - A string identifier specifying a velocity framework
                - A float, specifying a constant wave propagation speed. Currently, only this type can be used with dim=3
            attenuation_factor (float): The attenuation factor in the acoustic wave equation
            keep_full_data (bool): Whether keeping the full simulation or only the interrogable one
        """
        try:
            if dt < ddt:
                raise ValueError('dt should be >= ddt')
            self.size = size
            self.dx = dx
            self.nx = nx
            self.sx = sx
            self.ddt = ddt
            self.dt = dt
            self.nt = int(t / self.dt)
            self.ndt = int(self.nt * (self.dt / self.ddt))
            self.interrogators = interrogators
            self.interrogators_data = None
            self.attenuation_factor = attenuation_factor
            self.force_delay = np.random.random(size) * (t/2)
            self.amplitude_factor = (0.5 * np.random.random(size) + 0.25) * 2
            self.data = np.array([])
            self.keep_full_data = keep_full_data

        except ValueError as err:
            print(err)

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
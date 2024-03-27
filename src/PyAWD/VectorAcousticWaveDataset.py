# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
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

    def __init__(self, size: int, dx: float = 1000 / 128., nx: int = 128, sx: float = 1., ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, attenuation_factor: float = 0.5,
                 openmp: bool = False):
        """
        Args:
            size (int): The number of samples to generate in the dataset
            dx (float): The discretisation rate of the array
            nx (int): The discretisation size of the array
            sx (float): The sub-scaling factor of the array (0.5 means 1/2 values are returned)
            ddt (float): The time step used for the Operator solving iterations
            dt (float): The time step used for storing the wave propagation step (this should be higher than ddt)
            t (float): The simulations duration
            attenuation_factor (float): The attenuation factor in the acoustic wave equation
            openmp (bool): Use openmp optimization
        """
        super().__init__(size, dx, nx, sx, ddt, dt, t, interrogators, attenuation_factor, openmp=openmp)

    def save(self, filename: str):
        """
        Saves the dataset features to different files, starting with `filename`.
        This is intended to allow to retrieve the dataset by using "VectorAcousticWaveDataset2D().load(filename)`
        Args:
            filename (str): The base name of the files where the data will be saved
        """
        features = np.array([self.size, self.dx, self.nx, self.sx, self.ddt, self.dt, self.nt, self.ndt, self.attenuation_factor])
        np.save(filename+"_features.npy", features)
        np.save(filename+"_force_delays.npy", self.force_delay)
        np.save(filename+"_amplitude_factors.npy", self.amplitude_factor)
        np.save(filename+"_max_velocities.npy", self.max_velocities)
        np.save(filename+"_epicenters.npy", self.epicenters)
        np.save(filename+"_interrogators.npy", self.interrogators)
        np.save(filename+"_velocity_model.npy", self.velocity_model.data[:])

    @classmethod
    def load(cls, filename: str):
        """
        Loads a dataset that was saved through the `save()` method.
        Args:
            filename (str): The base name of the files where the data has been saved
        """
        features = np.load(filename+"_features.npy")
        dataset = cls(size=int(features[0]), dx=features[1], nx=int(features[2]), sx=features[3],
                      ddt=features[4], dt=features[5])
        dataset.nt = int(features[6])
        dataset.ndt = int(features[7])
        dataset.attenuation_factor = features[8]
        dataset.force_delay = np.load(filename+"_force_delays.npy")
        dataset.amplitude_factor = np.load(filename+"_amplitude_factors.npy")
        dataset.max_velocities = np.load(filename+"_max_velocities.npy")
        dataset.epicenters = np.load(filename+"_epicenters.npy")
        dataset.interrogators = [tuple(i) for i in np.load(filename+"_interrogators.npy")]
        dataset.velocity_model.data[:] = np.load(filename+"_velocity_model.npy")
        return dataset

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        raise NotImplementedError('This class is abstract')

    def solve_pde(self, idx: int):
        """
        Solves the Acoustic Wave Equation for the idx parameters.
        Returns:
            (numpy.ndarray): A numpy array containing the solutions for the `ndt` steps
        """
        raise NotImplementedError('This class is abstract')

    def plot_item(self, idx: int):
        """
        Plots the simulation of the $idx^{th}$ sample
        Args:
            idx (int): The number of the sample to plot
        """
        raise NotImplementedError('This class is abstract')

    def plot_interrogators_response(self, idx: int):
        """
        Plots the measurements taken by the interrogators for the $idx^{th}$ sample.
        Args:
            idx (int): The number of the sample to plot
        """
        raise NotImplementedError('This class is abstract')

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
        raise NotImplementedError('This class is abstract')

    def get_epicenter(self, idx):
        """
        Returns the epicenter of the $idx^{th}$ sample
        """
        return self.epicenters[idx]

    def get_max_velocity(self, idx):
        """
        Returns the maximal velocity of the $idx^{th}$ sample
        """
        return self.max_velocities[idx]

    def get_force_delay(self, idx):
        """
        Returns the delay before the external force occurring in the $idx^{th}$ sample
        """
        return self.force_delay[idx]

    def get_amplitude_factor(self, idx):
        """
        Returns the amplitude factor of the external force in the $idx^{th}$ sample
        """
        return self.amplitude_factor[idx]

    def __getitem__(self, idx):
        """
        Returns:
            (Tuple): The epicenter, the simulation of the `idx`th sample, the maximal speed of propagation of the
             propagation field, the delay before the external force application, the force amplitude factor and
             the interrogated data
        """
        raise NotImplementedError('This class is abstract')
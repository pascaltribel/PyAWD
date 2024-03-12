# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from typing import Tuple, List, Dict

import numpy as np
import devito as dvt
import matplotlib.colors
from matplotlib.colors import TABLEAU_COLORS
import torch.utils.data
from tqdm.auto import tqdm

from pyawd.GenerateVideo import generate_video
from pyawd.utils import *
from pyawd.Marmousi import *

COLORS = TABLEAU_COLORS

dvt.configuration['log-level'] = 'WARNING'


def solve_scalar_pde(grid: dvt.Grid, nx: int, ndt: int, ddt: float, epicenter: np.ndarray, velocity_model: dvt.Function) -> np.ndarray:
    """
    Solves the Acoustic Wave Equation for the input parameters
    Args:
        grid (dvt.Grid): A Devito Grid Object
        nx (int): The discretisation size of the array
        ndt (int): The number of iteration for which the result is stored
        ddt (float): The time step used for the Operator solving iterations
        epicenter (np.ndarray): The epicenter of the Ricker Wavelet at the beginning of the simulation
        velocity_model (devito.Function): The velocity field across which the wave propagates
    Returns:
        (numpy.ndarray): A numpy array containing the solutions for the `ndt` simulation steps
    """
    u = dvt.TimeFunction(name='u', grid=grid, space_order=2, save=ndt, time_order=2)
    u.data[:] = get_ricker_wavelet(nx, x0=epicenter[0], y0=epicenter[1])
    eq = dvt.Eq(u.dt2, (velocity_model ** 2) * (u.dx2 + u.dy2))
    stencil = dvt.solve(eq, u.forward)
    op = dvt.Operator(dvt.Eq(u.forward, stencil), opt='noop')
    op.apply(dt=ddt)
    return np.array(u.data)


class ScalarAcousticWaveDataset(torch.utils.data.Dataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field. This dataset can only build 2D simulations.
    """
    size: int
    """The number of samples to generate in the dataset"""
    nx: int
    """The discretisation size of the array (maximum size is currently 955)"""
    sx: float
    """The sub-scaling factor of the array (0.5 means $\\frac{1}{2}$ values are returned)"""
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
    epicenters: np.ndarray
    """The epicenter of each simulation"""
    data: np.ndarray
    """The simulations data"""
    cmap: matplotlib.colors.LinearSegmentedColormap
    """The colormap used for displaying the simulations"""
    def __init__(self, size: int, nx: int = 128, sx: float = 1., ddt: float = 0.01, dt: int = 2, t: float = 10,
                 interrogators: List[Tuple] = None):
        """
        Args:
            size (int): The number of samples to generate in the dataset
            nx (int): The discretisation size of the array (maximum size is currently 955)
            sx (float): The sub-scaling factor of the array (0.5 means $\\frac{1}{2}$ values are returned)
            ddt (float): The time step used for the Operator solving iterations
            dt (float): The time step used for storing the wave propagation step (this should be higher than ddt)
            t (float): The simulations duration
            interrogators (List[Tuple]): A list containing the coordinates of each interrogator
        """
        if interrogators is None:
            interrogators = [(0, 0)]
        try:
            if dt < ddt:
                raise ValueError('dt should be >= ddt')
            self.size = size
            self.nx = min(nx, 955)
            self.sx = sx
            self.ddt = ddt
            self.dt = dt
            self.nt = int(t / self.dt)
            self.ndt = int(self.nt * (self.dt / self.ddt))
            self.interrogators = interrogators
            self.interrogators_data = None

            self.grid = dvt.Grid(shape=(self.nx, self.nx), extent=(1., 1.))
            self.velocity_model = dvt.Function(name='c', grid=self.grid)
            self.velocity_model.data[:] = Marmousi(self.nx).get_data().T

            self.epicenters = np.random.randint(-self.nx // 2, self.nx // 2, size=(self.size, 2)).reshape(
                (self.size, 2))
            self.data = None
            self.generate_data()

            self.cmap = get_black_cmap()

        except ValueError as err:
            print(err)

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        self.data = []
        self.interrogators_data = {interrogator: [] for interrogator in self.interrogators}
        for i in tqdm(range(self.size)):
            data = solve_scalar_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[i], self.velocity_model)
            self.data.append(data[::int(self.ndt / self.nt)])
            for interrogator in self.interrogators:
                self.interrogators_data[interrogator].append(
                    data[:, interrogator[0] + (self.nx // 2), -interrogator[1] + (self.nx // 2)])
        self.data = np.array(self.data)

    def interrogate(self, idx: int, point: Tuple) -> np.ndarray:
        """
        Args:
            idx (int): the number of the sample to interrogate
            point (Tuple): the interrogator position as a Tuple
        Returns:
             (np.ndarray): The amplitude measurements for the interrogator at coordinates `point` for the `idx`th sample
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
            idx (int): the number of the sample to plot
        """
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        epicenter, item = self[idx]
        fig, ax = plt.subplots(ncols=self.nt, figsize=(self.nt * 3, 3))
        for i in range(self.nt):
            ax[i].imshow(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)].T,
                         vmin=np.min(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)]),
                         vmax=np.max(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)]), cmap="gray")
            x = ax[i].imshow(item[i * (item.shape[0] // self.nt)].T,
                             vmin=-np.max(np.abs(item[i * (item.shape[0] // self.nt):])),
                             vmax=np.max(np.abs(item[i * (item.shape[0] // self.nt):])),
                             cmap=self.cmap)
            for interrogator in self.interrogators:
                ax[i].scatter(interrogator[0] + (self.nx // 2), -interrogator[1] + (self.nx // 2), marker="1",
                              color=colors[interrogator])
            ax[i].set_title("t = " + str(i * (item.shape[0] // self.nt) * self.dt) + "s")
            ax[i].axis("off")
            fig.colorbar(x)
        plt.tight_layout()
        plt.show()

    def plot_interrogators_response(self, idx: int):
        """
        Plots the measurements taken by the interrogators for the $idx^{th}$ sample.
        Args:
            idx (int): the number of the sample to plot
        """
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        for i in self.interrogators:
            plt.plot(np.arange(0, self.ndt * self.ddt, self.ddt), self.interrogate(idx, i), color=colors[i])
        plt.xlabel("time (s)")
        plt.ylabel("Amplitude")
        plt.legend([str(i) for i in self.interrogators])
        plt.title("Amplitude measurement for each interrogator")

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
        u = solve_scalar_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[idx], self.velocity_model)
        generate_video(u[::self.ndt // nb_images], self.interrogators,
                       {i: self.interrogators_data[i][idx][::self.ndt // nb_images] for i in self.interrogators},
                       filename, nx=self.nx, dt=self.ndt * self.ddt / nb_images, c=self.velocity_model, verbose=True)

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

    def __getitem__(self, idx: int):
        """
        Returns:
            (Tuple): The epicenter and the simulation of the `idx`th sample
        """
        return self.epicenters[idx], self.data[idx][:, ::int(1 / self.sx), ::int(1 / self.sx)]

# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from typing import Tuple, List, Dict

import numpy as np
import devito as dvt
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import TABLEAU_COLORS

from tqdm.auto import tqdm
import torch.utils.data
from pyawd.GenerateVideo import generate_quiver_video, generate_density_video
from pyawd.utils import get_black_cmap, create_explosive_source
from pyawd.Marmousi import Marmousi

COLORS = TABLEAU_COLORS
dvt.configuration['log-level'] = "WARNING"


def solve_vector_pde(grid: dvt.Grid, nx: int, ndt: int, ddt: float, epicenter: np.ndarray, velocity_model: dvt.Function,
                     max_velocity: np.ndarray, f_delay: np.ndarray, amplitude_factor: np.ndarray, dim: int = 2):
    """
    Solves the Acoustic Wave Equation for the input parameters.
    The returned array has its both component in the shape
            →→x→→
            ↓
            y
            ↓
    Args:
        grid (devito.Grid): A Devito Grid Object
        nx (int): The discretisation size of the array
        ndt (int): The number of iteration for which the result is stored
        ddt (float): The time step used for the Operator solving iterations
        epicenter (numpy.ndarray): The epicenter of the Ricker Wavelet at the beginning of the simulation
        velocity_model (devito.Function): The velocity field across which the wave propagates
        max_velocity (np.ndarray): The maximal velocity in the velocity model
        f_delay (np.ndarray): The time delay before starting the external force
        amplitude_factor (np.ndarray): The amplitude factor of the external force
        dim (int): The number of dimensions of the simulations (2 or 3)
    Returns:
        (numpy.ndarray): A numpy array containing the solutions for the `ndt` steps
    """
    u = dvt.VectorTimeFunction(name='u', grid=grid, space_order=2, save=ndt, time_order=2)

    u[0].data[:] = 1e-5 * (np.random.random(u[0].data[:].shape) - 0.5)
    u[1].data[:] = 1e-5 * (np.random.random(u[1].data[:].shape) - 0.5)
    if dim == 3:
        u[2].data[:] = 1e-5 * (np.random.random(u[1].data[:].shape) - 0.5)
    f = dvt.VectorTimeFunction(name='f', grid=grid, space_order=1, save=ndt, time_order=1)
    s_t = amplitude_factor * np.exp(-ddt * (np.arange(ndt) - (f_delay / ddt)) ** 2)
    if dim == 2:
        s_x, s_y = create_explosive_source(nx, x0=int(epicenter[0]), y0=int(epicenter[1]))
    else:
        s_x, s_y, s_z = create_explosive_source(nx, x0=int(epicenter[0]), y0=int(epicenter[1]), z0=int(epicenter[2]),
                                                dim=dim)
    s_x_t = np.array([s_x * s_t[t] for t in range(len(s_t))])
    s_y_t = np.array([s_y * s_t[t] for t in range(len(s_t))])
    f[0].data[:] = s_x_t
    f[1].data[:] = s_y_t
    if dim == 3:
        s_z_t = np.array([s_z * s_t[t] for t in range(len(s_t))])
        f[2].data[:] = s_z_t
    eq = dvt.Eq(u.dt2, f + ((max_velocity * velocity_model) ** 2) * u.laplace)
    stencil = dvt.solve(eq, u.forward)
    op = dvt.Operator(dvt.Eq(u.forward, stencil), opt='noop')
    op.apply(dt=ddt)
    return np.array([u[i].data for i in range(len(u))])


class VectorAcousticWaveDataset(torch.utils.data.Dataset):
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
    epicenters: np.ndarray
    """The epicenter of each simulation"""
    force_delay: np.ndarray
    """The delay of apparition of the external force for each simulation"""
    amplitude_factor: np.ndarray
    """The amplitude factor to multiply the external force with"""
    data: np.ndarray
    """The simulations data"""
    cmap: matplotlib.colors.LinearSegmentedColormap
    """The colormap used for displaying the simulations"""
    def __init__(self, size: int, dx: float = 1000/128., nx: int = 128, sx: float = 1., dim: int = 2, ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, velocity_model: str | int = 300):
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
            velocity_model (Tuple | int): either:
                - A tuple (name, max_value) specifying the maximum wave propagation speed in the specified framework
                - An integer, specifying a constant wave propagation speed. Currently, only this type can be used with dim=3
        """
        if interrogators is None:
            interrogators = [tuple(0 for _ in range(dim))]
        try:
            if dt < ddt:
                raise ValueError('dt should be >= ddt')
            self.size = size
            self.dx = dx
            self.nx = nx
            self.sx = sx
            self.dim = dim
            self.ddt = ddt
            self.dt = dt
            self.nt = int(t / self.dt)
            self.ndt = int(self.nt * (self.dt / self.ddt))
            self.interrogators = interrogators
            self.interrogators_data = None

            self.grid = dvt.Grid(shape=tuple(self.nx for _ in range(self.dim)), extent=tuple(self.dx*self.nx for _ in range(self.dim)))
            self.velocity_model = dvt.Function(name='c', grid=self.grid)
            if velocity_model == "Marmousi":
                self.velocity_model.data[:] = Marmousi(self.nx).get_data() * 10
                self.max_velocities = (np.random.random(size) * 0.5 + 0.5) * 400
            elif isinstance(velocity_model, float) or isinstance(velocity_model, int) :
                self.velocity_model.data[:] = velocity_model
                self.max_velocities = np.ones(size)
            self.epicenters = np.random.randint(-self.nx // 2, self.nx // 2, size=(self.size, self.dim)).reshape(
                (self.size, self.dim))
            self.force_delay = np.random.random(size) * (t/2)
            self.amplitude_factor = (0.5 * np.random.random(size) + 0.25) * 2
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
            data = solve_vector_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[i], self.velocity_model,
                                    self.max_velocities[i], self.force_delay[i], self.amplitude_factor[i], self.dim)
            self.data.append(data[:, ::int(self.ndt / self.nt)])
            for interrogator in self.interrogators:
                if self.dim == 2:
                    self.interrogators_data[interrogator].append(data[:, :, interrogator[0] +
                                                                 (self.nx // 2), interrogator[1] +
                                                                 (self.nx // 2)])
                else:
                    self.interrogators_data[interrogator].append(data[:, :, interrogator[0] +
                                                                 (self.nx // 2), interrogator[1] +
                                                                 (self.nx // 2), interrogator[2] +
                                                                 (self.nx // 2)])
        self.data = np.array(self.data)

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
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        epicenter, item, max_velocity, f_delay, amplitude_factor = self[idx]
        fig = plt.figure(figsize=(self.nt * 5, 5))
        if self.dim == 2:
            ax = []
            a, b = np.meshgrid(np.arange(self.nx), np.arange(self.nx))
            for i in range(self.nt):
                ax.append(fig.add_subplot(1, self.nt, i+1))
                ax[i].imshow(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)],
                             vmin=np.min(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)]),
                             vmax=np.max(self.velocity_model.data[::int(1 / self.sx), ::int(1 / self.sx)]), cmap="gray")
                ax[i].quiver(a, b, item[0][i * (item.shape[1] // self.nt)], -item[1][i * (item.shape[1] // self.nt)],
                             scale=0.25)
                for interrogator in self.interrogators:
                    ax[i].scatter(interrogator[0] + (self.nx // 2), interrogator[1] + (self.nx // 2), marker="1",
                                  color=colors[interrogator])
                ax[i].set_title("t = " + str(i * (item.shape[1] // self.nt) * self.dt) + "s, \nVelocity factor = " +
                                str(max_velocity)[:5] + ", \nForce delay = " + str(f_delay)[:4] +
                                ", \nAmplitude factor = " + str(amplitude_factor)[:4])
                ax[i].axis("off")
        else:
            ax = []
            a, b, c = np.meshgrid(np.arange(self.nx), np.arange(self.nx), np.arange(self.nx))
            for i in range(self.nt):
                ax.append(fig.add_subplot(1, self.nt, i + 1, projection='3d'))
                ax[i].quiver(a, b, c, item[0][i * (item.shape[1] // self.nt)], -item[1][i * (item.shape[1] // self.nt)],
                             item[2][i * (item.shape[1] // self.nt)], arrow_length_ratio=0.25)
                for interrogator in self.interrogators:
                    ax[i].scatter(interrogator[0] + (self.nx // 2), interrogator[1] + (self.nx // 2),
                                  interrogator[2] + (self.nx // 2), marker="1",
                                  color=colors[interrogator])
                ax[i].set_title("t = " + str(i * (item.shape[1] // self.nt) * self.dt) + "s, \nVelocity factor = " +
                                str(max_velocity)[:5] + ", \nForce delay = " + str(f_delay)[:4] +
                                ", \nAmplitude factor = " + str(amplitude_factor)[:4])
                ax[i].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.tight_layout()
        plt.show()

    def plot_interrogators_response(self, idx: int):
        """
        Plots the measurements taken by the interrogators for the $idx^{th}$ sample.
        Args:
            idx (int): The number of the sample to plot
        """
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        fig, ax = plt.subplots(ncols=len(self.interrogators), figsize=(len(self.interrogators) * 5, 5))
        y_lims = []
        for i in range(len(self.interrogators)):
            data = self.interrogate(idx, self.interrogators[i])
            y_lims += [np.min(data), np.max(data)]
            for j in range(data.shape[0]):
                if len(self.interrogators) == 1:
                    ax.plot(np.arange(0, self.ndt * self.ddt, self.ddt), data[j], linestyle=['-', '--', ':'][j],
                            color=colors[self.interrogators[i]])
                else:
                    ax[i].plot(np.arange(0, self.ndt * self.ddt, self.ddt), data[j], linestyle=['-', '--', ':'][j],
                               color=colors[self.interrogators[i]])
            if len(self.interrogators) == 1:
                ax.legend(["Abscissa", "Ordinate", "Applicate"][:self.dim])
                ax.set_title(str(self.interrogators[i]))
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_ylim([np.min(data), np.max(data)])
            else:
                ax[i].legend(["Abscissa", "Ordinate", "Applicate"][:self.dim])
                ax[i].set_title(str(self.interrogators[i]))
                ax[i].set_xlabel("time (s)")
                ax[i].set_ylabel("Amplitude")
        if len(self.interrogators) > 1:
            for i in range(len(self.interrogators)):
                ax[i].set_ylim([np.min(y_lims), np.max(y_lims)])
            fig.suptitle("Velocity factor = " + str(self.max_velocities[idx])[:5] + "\nForce delay = " + str(
                self.force_delay[idx])[:4] + "\nAmplitude factor = " + str(self.amplitude_factor[idx])[:4] +
                         "\nEpicenter = " + str(self.epicenters[idx]))
            plt.tight_layout()

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
        u = solve_vector_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[idx], self.velocity_model,
                             self.max_velocities[idx], self.force_delay[idx], self.amplitude_factor[idx], self.dim)
        if self.dim == 2:
            generate_quiver_video(u[0][::self.ndt // nb_images], u[1][::self.ndt // nb_images], self.interrogators,
                                  {i: self.interrogate(idx, i)[:, ::self.ndt // nb_images] for i in self.interrogators},
                                  filename, nx=self.nx, dt=self.ndt * self.ddt / nb_images, c=self.velocity_model,
                                  max_velocity=self.max_velocities[idx], dim=self.dim, verbose=True)
        else:
            generate_density_video(u[0][::self.ndt // nb_images], u[1][::self.ndt // nb_images],
                                   u[2][::self.ndt // nb_images], self.interrogators,
                                   {i: self.interrogate(idx, i)[:, ::self.ndt // nb_images] for i in self.interrogators},
                                   filename, nx=self.nx, dt=self.ddt * (self.ndt // nb_images), dx=self.dx)

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
        if self.dim == 2:
            return (self.epicenters[idx], self.data[idx][:, :, ::int(1 / self.sx), ::int(1 / self.sx)],
                    self.max_velocities[idx], self.force_delay[idx], self.amplitude_factor[idx])
        else:
            return self.epicenters[idx], self.data[idx][:, :, ::int(1 / self.sx), ::int(1 / self.sx), ::int(1 / self.sx)], \
            self.max_velocities[idx], self.force_delay[idx], self.amplitude_factor[idx]

# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from memory_profiler import profile
from typing import Tuple, List, Dict, Union

import numpy as np
import devito as dvt
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import TABLEAU_COLORS

from tqdm.auto import tqdm
from pyawd import VectorAcousticWaveDataset
from pyawd.GenerateVideo import generate_quiver_video, generate_density_video
from pyawd.utils import create_explosive_source, create_inverse_distance_matrix
from pyawd.Marmousi import Marmousi

COLORS = TABLEAU_COLORS
dvt.configuration['log-level'] = "WARNING"


class VectorAcousticWaveDataset3D(VectorAcousticWaveDataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field
    """

    def __init__(self, size: int, dx: float = 1000/32., nx: int = 32, sx: float = 1., ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, velocity_model: Union[
                 str, float] = 300., attenuation_factor: float = 0.5, keep_full_data: bool = True):
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
        if interrogators is None:
            interrogators = [tuple(0 for _ in range(3))]
        super().__init__(size=size, dx=dx, nx=nx, sx=sx, ddt=ddt, dt=dt, t=t, interrogators=interrogators,
                         velocity_model=velocity_model, attenuation_factor=attenuation_factor, keep_full_data=keep_full_data)

        self.grid = dvt.Grid(shape=tuple(self.nx for _ in range(3)), extent=tuple(self.dx*self.nx for _ in range(3)))
        self._u = dvt.VectorTimeFunction(name='u', grid=self.grid, space_order=2, save=self.ndt, time_order=2)
        self._f = dvt.VectorTimeFunction(name='f', grid=self.grid, space_order=1, save=self.ndt, time_order=1)
        self.velocity_model = dvt.Function(name='c', grid=self.grid)
        if velocity_model == "Marmousi":
            print("Marmousi model is only available in 2D")
        self._display_velocity_model = False
        self.velocity_model.data[:] = velocity_model
        self.max_velocities = np.ones(size)
        self.epicenters = np.random.randint(-self.nx // 2, self.nx // 2, size=(self.size, 3)).reshape((self.size, 3))
        self.generate_data()

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        self.data = []
        self.interrogators_data = {interrogator: [] for interrogator in self.interrogators}
        for idx in tqdm(range(self.size)):
            data = self.solve_pde(idx)
            if self.keep_full_data:
                self.data.append(data[:, ::int(self.ndt / self.nt)])
            for interrogator in self.interrogators:
                self.interrogators_data[interrogator].append(data[:, :, interrogator[0] +
                                                             (self.nx // 2), interrogator[1] +
                                                             (self.nx // 2), interrogator[2] +
                                                             (self.nx // 2)])
        self.data = np.array(self.data)

    def solve_pde(self, idx: int):
        """
        Solves the Acoustic Wave Equation for the idx parameters.
        Returns:
            (numpy.ndarray): A numpy array containing the solutions for the `ndt` steps
        """
        self._u[0].data[:] = 1e-5 * (np.random.random(self._u[0].data[:].shape) - 0.5)
        self._u[1].data[:] = 1e-5 * (np.random.random(self._u[1].data[:].shape) - 0.5)
        s_t = self.amplitude_factor[idx] * np.exp(-self.ddt * (np.arange(self.ndt) - (self.force_delay[idx] / self.ddt)) ** 2)
        self._u[2].data[:] = 1e-5 * (np.random.random(self._u[2].data[:].shape) - 0.5)
        s_x, s_y, s_z = create_explosive_source(self.nx, x0=int(self.epicenters[idx][0]), y0=int(self.epicenters[idx][1]),
                                                z0=int(self.epicenters[idx][2]), dim=3)
        self._f[0].data[:] = (np.tile(s_x, (s_t.shape[0], 1, 1, 1)) * s_t[:, None, None, None])
        self._f[1].data[:] = (np.tile(s_y, (s_t.shape[0], 1, 1, 1)) * s_t[:, None, None, None])
        self._f[2].data[:] = (np.tile(s_z, (s_t.shape[0], 1, 1, 1)) * s_t[:, None, None, None])
        op = dvt.Operator(dvt.Eq(self._u.forward,
                                 dvt.solve(dvt.Eq(self._u.dt2, self._f +
                                                  ((self.max_velocities[idx] * self.velocity_model) ** 2)
                                                  * self._u.laplace - self.attenuation_factor*self._u.dt),
                                           self._u.forward)), opt=('advanced'))
        op.apply(dt=self.ddt)
        return np.array([self._u[i].data for i in range(self._u.shape[0])])

    def plot_item(self, idx: int):
        """
        Plots the simulation of the $idx^{th}$ sample
        Args:
            idx (int): The number of the sample to plot
        """
        if not self.keep_full_data:
            print("Full data has not been stored.")
            return
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        epicenter, item, max_velocity, f_delay, amplitude_factor = self[idx]
        fig = plt.figure(figsize=(self.nt * 5, 5))
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
                ax.legend(["Abscissa", "Ordinate", "Applicate"])
                ax.set_title(str(self.interrogators[i]))
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_ylim([np.min(data), np.max(data)])
            else:
                ax[i].legend(["Abscissa", "Ordinate", "Applicate"])
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
        u = self.solve_pde(idx)
        generate_density_video(u[0][::self.ndt // nb_images], u[1][::self.ndt // nb_images],
                               u[2][::self.ndt // nb_images], self.interrogators,
                               {i: u[:, ::self.ndt // nb_images, i[0] + (self.nx // 2), i[1] + (self.nx // 2), i[2] + (self.nx // 2)]
                                for i in self.interrogators},
                               filename, nx=self.nx, dt=self.ddt * (self.ndt // nb_images), dx=self.dx)

    def __getitem__(self, idx):
        """
        Returns:
            (Tuple): The epicenter, the simulation of the `idx`th sample, the maximal speed of propagation of the
             propagation field, the delay before the external force application and the force amplitude factor
        """
        return (self.epicenters[idx], self.data[idx][:, :, ::int(1 / self.sx), ::int(1 / self.sx), ::int(1 / self.sx)],
                self.max_velocities[idx], self.force_delay[idx], self.amplitude_factor[idx])
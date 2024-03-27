# pyawd - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be
from typing import Tuple, List, Union

import numpy as np
import devito as dvt
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import torch

from pyawd import VectorAcousticWaveDataset, VelocityModel3D
from pyawd.GenerateVideo import generate_density_video
from pyawd.utils import create_explosive_source

COLORS = TABLEAU_COLORS
dvt.configuration['log-level'] = "WARNING"


class VectorAcousticWaveDataset3D(VectorAcousticWaveDataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field
    """

    def __init__(self, size: int, dx: float = 1000/32., nx: int = 32, sx: float = 1., ddt: float = 0.01,
                 dt: float = 2, t: float = 10, interrogators: List[Tuple] = None, velocity_model: Union[
                 str, float] = 300., attenuation_factor: float = 0.5, openmp: bool = False):
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
            openmp (bool): Use openmp optimization
        """
        if interrogators is None:
            interrogators = [tuple(0 for _ in range(3))]
        super().__init__(size=size, dx=dx, nx=nx, sx=sx, ddt=ddt, dt=dt, t=t, interrogators=interrogators,
                         attenuation_factor=attenuation_factor, openmp=openmp)

        self.grid = dvt.Grid(shape=tuple(self.nx for _ in range(3)), extent=tuple(self.dx*self.nx for _ in range(3)))
        self._u = dvt.VectorTimeFunction(name='u', grid=self.grid, space_order=2, save=self.ndt, time_order=2)
        self._f = dvt.VectorTimeFunction(name='f', grid=self.grid, space_order=1, save=self.ndt, time_order=1)
        self._velocity_model = VelocityModel3D(self.nx)
        self.velocity_model = dvt.Function(name='c', grid=self.grid)
        if velocity_model == "Marmousi":
            raise NotImplementedError("Marmousi model is only available in 2D")
        self._velocity_model.set_value(velocity_model)
        self._display_velocity_model = False
        self.velocity_model.data[:] = self._velocity_model.get_data()
        self.max_velocities = np.ones(size)
        self.epicenters = np.random.randint(-self.nx // 2, self.nx // 2, size=(self.size, 3)).reshape((self.size, 3))

    def solve_pde(self, idx: int) -> np.ndarray:
        """
        Solves the Acoustic Wave Equation for the idx parameters.
        Returns:
            (numpy.ndarray): A numpy array containing the solutions for the `ndt` steps
        """
        self._u[0].data[:] = 1e-5 * (np.random.random(self._u[0].data[:].shape) - 0.5)
        self._u[1].data[:] = 1e-5 * (np.random.random(self._u[1].data[:].shape) - 0.5)
        self._u[2].data[:] = 1e-5 * (np.random.random(self._u[2].data[:].shape) - 0.5)
        self.s_t = self.amplitude_factor[idx] * np.exp(-self.ddt * (np.arange(self.ndt) -
                                                                    (self.force_delay[idx] / self.ddt)) ** 2)
        self.s_x, self.s_y, self.s_z = create_explosive_source(self.nx, x0=int(self.epicenters[idx][0]),
                                                               y0=int(self.epicenters[idx][1]),
                                                               z0=int(self.epicenters[idx][2]), dim=3)
        self.reps = (self.s_t.shape[0], 1, 1, 1)
        self.s_t_2 = self.s_t[:, None, None, None]
        self._f[0].data[:] = np.tile(self.s_x, self.reps) * self.s_t_2
        self._f[1].data[:] = np.tile(self.s_y, self.reps) * self.s_t_2
        self._f[2].data[:] = np.tile(self.s_z, self.reps) * self.s_t_2
        self.op = dvt.Operator(dvt.Eq(self._u.forward,
                                 dvt.solve(dvt.Eq(self._u.dt2, self._f +
                                                  ((self.max_velocities[idx] * self.velocity_model) ** 2)
                                                  * self._u.laplace - self.attenuation_factor*self._u.dt),
                                           self._u.forward)), opt=('advanced', {'openmp': self.openmp}))
        self.op.apply(dt=self.ddt)
        return np.array([self._u[i].data for i in range(self._u.shape[0])])

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
        item, _ = self[idx]
        epicenter = self.epicenters[idx]
        max_velocity = self.max_velocities[idx]
        f_delay = self.force_delay[idx]
        amplitude_factor = self.amplitude_factor[idx]
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
                            ", \nAmplitude factor = " + str(amplitude_factor)[:4] + "\nEpicenter = " + str(epicenter))
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
        _, full_data = self[idx]
        for i in range(len(self.interrogators)):
            data = full_data[self.interrogators[i]]
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
        plt.show()

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
                               {i: u[:, ::self.ndt // nb_images, i[0] +
                                (self.nx // 2), i[1] + (self.nx // 2), i[2] + (self.nx // 2)]
                                for i in self.interrogators},
                               filename, nx=self.nx, dt=self.ddt * (self.ndt // nb_images), dx=self.dx)

    def __getitem__(self, idx) -> Tuple:
        """
        Returns:
            (Tuple): The epicenter, the simulation of the `idx`th sample, the maximal speed of propagation of the
             propagation field, the delay before the external force application, the force amplitude factor and
             the interrogated data
        """
        data = self.solve_pde(idx)
        return (torch.Tensor(data[:, :int(self.ndt / self.nt),
                             ::int(1 / self.sx), ::int(1 / self.sx), ::int(1 / self.sx)]),
                {i: data[:, :, i[0] + (self.nx // 2), i[1] + (self.nx // 2), i[2] + (self.nx // 2)]
                 for i in self.interrogators})
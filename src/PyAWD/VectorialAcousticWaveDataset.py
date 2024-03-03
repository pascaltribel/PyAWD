# PyAWD - AcousticWaveDataset
# Tribel Pascal - pascal.tribel@ulb.be

import numpy as np
import devito as dvt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm.auto import tqdm

from PyAWD.GenerateVideo import generate_quiver_video
from PyAWD.utils import *
from PyAWD.Marmousi import *

COLORS = mcolors.TABLEAU_COLORS

dvt.logger.set_log_level('WARNING')

def solve_vectorial_pde(grid, nx, ndt, ddt, epicenter, velocity_model):
    """
    Solves the Acoustic Wave Equation for the input parameters
    Arguments:
        - grid: a Devito Grid Object
        - nx: the discretization size of the array
        - ndt: the number of iteration for which the result is stored
        - ddt: the time step used for the Operator solving iterations
        - epicenter: the epicenter of the Ricker Wavelet at the beginning of the simulation
        - velocity_model: the velocity field across which the wave propagates
    Returns:
        - u: a Devito TimeFunction containing the solutions for the `ndt` steps
    Remarks:
        - The return array has its both component in the shape
            →→x→→
            ↓
            y
            ↓
    """
    u = dvt.VectorTimeFunction(name='u', grid=grid, space_order=2, save=ndt, time_order=2)
    f = dvt.VectorTimeFunction(name='f', grid=grid, space_order=1, save=ndt, time_order=1)
    s_x, s_y = create_explosive_source(nx, x0=epicenter[0], y0=epicenter[1])
    s_t = np.exp(-(ddt)*(np.arange(ndt)-(ndt//10))**2)
    s_x_t = np.array([s_x*s_t[t] for t in range(len(s_t))])
    s_y_t = np.array([s_y*s_t[t] for t in range(len(s_t))])
    f[0].data[:] = s_x_t
    f[1].data[:] = s_y_t
    eq = dvt.Eq(u.dt2, f+(velocity_model**2)*(u.laplace))
    stencil = dvt.solve(eq, u.forward)
    op = dvt.Operator(dvt.Eq(u.forward, stencil), opt='noop')
    op.apply(dt=ddt)
    return np.array([u[0].data, u[1].data])

class VectorialAcousticWaveDataset(torch.utils.data.Dataset):
    """
    A Pytorch dataset containing acoustic waves propagating in the Marmousi velocity field.
    Arguments:
        - size: the number of samples to generate in the dataset
        - nx: the discretization size of the array (maximum size is currently 955)
        - sx: the sub-scaling factor of the array (0.5 means 1/2 values are returned)
        - ddt: the time step used for the Operator solving iterations
        - dt: the time step used for storing the wave propagation step (this should be higher than ddt)
        - t: the simulations duration
        - velocity_model: either:
            - a tuple (name, max_value) specifying the maximum wave propagation speed in the specified framework
            - an integer, specifying a constant wave propagation speed
    """
    def __init__(self, size, nx=128, sx=1., ddt=0.01, dt=2, t=10, interrogators=[(0, 0)], velocity_model=("Marmousi", 100)):
        try:
            if dt < ddt:
                raise ValueError('dt should be >= ddt')
            self.size = size
            self.nx = min(nx, 955)
            self.sx = sx
            self.ddt = ddt
            self.dt = dt
            self.nt = int(t/self.dt)
            self.ndt = int(self.nt*(self.dt/self.ddt))
            self.interrogators = interrogators
            
            self.grid = dvt.Grid(shape=(self.nx, self.nx), extent=(1000., 1000.))
            self.velocity_model = dvt.Function(name='c', grid=self.grid)
            if type(velocity_model) == tuple:
                self.velocity_model.data[:] = Marmousi(self.nx).get_data()
                self.velocity_model.data[:] *= (100/np.max(self.velocity_model.data[:]))
            elif type(velocity_model) == float or type(velocity_model) == int:
                self.velocity_model.data[:] = velocity_model
            self.epicenters = np.random.randint(-self.nx//2, self.nx//2, size=(self.size, 2)).reshape((self.size, 2))
            self.generate_data()

            self.cmap = get_black_cmap()
            
        except ValueError as err:
            print(err)

    def generate_data(self):
        """
        Generates the dataset content by solving the Acoustic Wave PDE for each of the `epicenters`
        """
        self.data = []
        self.interrogators_data = {interrogator:[] for interrogator in self.interrogators}
        for i in tqdm(range(self.size)):
            data = solve_vectorial_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[i], self.velocity_model)
            self.data.append(data[:, ::int(self.ndt/self.nt)])
            for interrogator in self.interrogators:
                self.interrogators_data[interrogator].append(data[:, :, interrogator[1]+(self.nx//2), interrogator[0]+(self.nx//2)])
        self.data = np.array(self.data)

    def interrogate(self, idx, point):
        """
        Returns the amplitude measurements for the interrogator at coordinates `point` for the `idx`th sample. 
        Arguments:
            - idx: the number of the sample to interrogate
            - point: the interrogator position as a Tuple
        """
        if point not in self.interrogators_data:
            print("Error: the interrogated point is not interrogable.")
            print("Available interrogable points:", list(self.interrogators_data.keys()))
        else:
            return self.interrogators_data[point][idx]

    def plot_item(self, idx):
        """
        Plots the simulation of the idx^th sample
        Arguments:
            - idx: the number of the sample to plot
        """
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        epicenter, item = self[idx]
        fig, ax = plt.subplots(1, self.nt, figsize=(self.nt*3, 3))
        a, b = np.meshgrid(np.arange(self.nx), np.arange(self.nx))
        for i in range(self.nt):
            ax[i].imshow(self.velocity_model.data[::int(1/self.sx), ::int(1/self.sx)], vmin=np.min(self.velocity_model.data[::int(1/self.sx), ::int(1/self.sx)]), vmax=np.max(self.velocity_model.data[::int(1/self.sx), ::int(1/self.sx)]), cmap="gray")
            ax[i].quiver(a, b, item[0][i*(item.shape[1]//self.nt)], -item[1][i*(item.shape[1]//self.nt)], scale=0.25)
            for interrogator in self.interrogators:
                ax[i].scatter(interrogator[0]+(self.nx//2), interrogator[1]+(self.nx//2), marker="1", color=colors[interrogator])
            ax[i].set_title("t = " + str(i*(item.shape[1]//self.nt)*self.dt) + "s")
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()
        print(colors)
        
    def plot_interrogators_response(self, idx):
        """
        Plots the measurements taken by the interrogators for the idx^th sample.
        Arguments:
            - idx: the number of the sample to plot
        """
        colors = {}
        i = 0
        for interrogator in self.interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
        fig, ax = plt.subplots(1,len(self.interrogators), figsize=(len(self.interrogators)*5, 5))
        y_lims = []
        for i in range(len(self.interrogators)):
            data = self.interrogate(idx, self.interrogators[i])
            y_lims += [np.min(data), np.max(data)]
            for j in range(data.shape[0]):
                if len(self.interrogators) == 1:
                    ax.plot(np.arange(0, self.ndt*self.ddt, self.ddt), data[j], linestyle=['-', '--', '-.'][j], color=colors[self.interrogators[i]])
                else:
                    ax[i].plot(np.arange(0, self.ndt*self.ddt, self.ddt), data[j], linestyle=['-', '--', '-.'][j], color=colors[self.interrogators[i]])
            if len(self.interrogators) == 1:
                ax.legend(["Horizontal", "Vertical"])
                ax.set_title(str(self.interrogators[i]))
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_ylim([np.min(data), np.max(data)])
            else:
                ax[i].legend(["Horizontal", "Vertical"])
                ax[i].set_title(str(self.interrogators[i]))
                ax[i].set_xlabel("time (s)")
                ax[i].set_ylabel("Amplitude")
        if len(self.interrogators) > 1:
            for i in range(len(self.interrogators)):
                ax[i].set_ylim([np.min(y_lims), np.max(y_lims)])
            plt.tight_layout()

    def generate_video(self, idx, filename, nb_images):
        """
        Generates a video representing the simulation of the idx^th sample propagation
        Arguments:
            - idx: the number of the sample to simulate in the video
            - filename: the name of the video output file (without extension)
                        The video will be stored in a file called `filename`.mp4
            - nb_images: the number of frames used to generate the video. This should be an entire divider of the number of points computed when applying the solving operator
        """
        u = solve_vectorial_pde(self.grid, self.nx, self.ndt, self.ddt, self.epicenters[idx], self.velocity_model)
        generate_quiver_video(u[0][::self.ndt//(nb_images)], u[1][::self.ndt//(nb_images)], self.interrogators, {i: self.interrogate(idx, i)[:, ::self.ndt//(nb_images)] for i in self.interrogators}, filename, nx=self.nx, dt=self.ndt*self.ddt/(nb_images), c=self.velocity_model, verbose=True)

    def set_scaling_factor(self, sx):
        """
        Fixes a new scaling factor (0.5 means 1/2 values are returned). It should be <= 1.
        Arguments:
            - sx: the new scaling factor
        """
        if sx <= 1.:
            self.sx = sx
        else:
            print("The scaling factor should be lower or equal to 1.")
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.epicenters[idx], self.data[idx][:, :, ::int(1/self.sx), ::int(1/self.sx)]
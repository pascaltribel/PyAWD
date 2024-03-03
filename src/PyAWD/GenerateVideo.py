# PyAWD - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
import numpy as np
from subprocess import call
from os import remove, chdir
from tqdm.notebook import tqdm

from PyAWD.utils import *

COLORS = mcolors.TABLEAU_COLORS

def generate_video(img, interrogators=None, interrogators_data=None, name="test", nx=32, dt=0.01, c=[], verbose=False):
    """
    Generates a video from a sequence of images.
    Arguments:
        - img: a list of 2d np.arrays representing the images
        - interrogators: a list of the interrogators positions
        - name: the name of the output file (without extension)
        - nx: the size of the images
        - dt: the time interval between each images
        - c: the background image representing the velocity field
        - verbose: if True, displays logging informations
    """
    colors = {}
    i = 0
    if interrogators:
        for interrogator in interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
    if verbose:
        print("Generating", len(img), "images.")
    for i in tqdm(range(len(img))):
        if interrogators:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})
            if c != []:
                ax[0].imshow(c.data.T, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
            im = ax[0].imshow(img[i].T, cmap=get_black_cmap(), vmin=-np.max(np.abs(img[i:])), vmax=np.max(np.abs(img[i:])))
            plt.colorbar(im, shrink=0.75, ax=ax[0])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw={'width_ratios': [1]})
            if c != []:
                ax.imshow(c.data.T, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
            im = ax.imshow(img[i].T, cmap=get_black_cmap(), vmin=-np.max(np.abs(img[i:])), vmax=np.max(np.abs(img[i:])))
            ax.axis('off')
            plt.colorbar(im, shrink=0.75, ax=ax)
        if interrogators:
            for interrogator in interrogators:
                ax[0].scatter(interrogator[0]+(nx//2), -interrogator[1]+(nx//2), marker="1", color=colors[interrogator])
                ax[1].plot(np.arange(0, len(img)*dt, dt)[:i+1], interrogators_data[interrogator][:i+1], color=colors[interrogator])
            ax[1].set_xlabel("Time")
            ax[1].set_ylabel("Amplitude")
            ax[1].legend([str(i) for i in interrogators_data])
            ax[1].set_ylim((np.min(np.array(list(interrogators_data.values()))), np.max(np.array(list(interrogators_data.values())))))
            ax[0].axis('off')
        plt.title("t = " + str(dt*i)[:4] + "s")
        plt.savefig(name + "%02d.png" % i, dpi=250)
        plt.close()
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', str(int(1/dt)), '-i', name + '%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name + ".mp4", '-y'
    ])
    for file_name in glob(name+"*.png"):
        remove(file_name)

def generate_quiver_video(quiver_x, quiver_y, interrogators=None, interrogators_data=None, name="test", nx=32, dt=0.01, c=[], verbose=False):
    """
    Generates a video from a sequence of images.
    Arguments:
        - quiver_x: a list 2d np.arrays representing the arrows x directions
        - quiver_y: a list 2d np.arrays representing the arrows y directions
        - interrogators: a list of the interrogators positions
        - interrogators_data: a list containing the interrogators responses
        - name: the name of the output file (without extension)
        - nx: the size of the images
        - dt: the time interval between each images
        - c: the background image representing the velocity field
        - verbose: if True, displays logging informations
    """
    nu_x = np.max(quiver_x)
    nu_y = np.max(quiver_y)
    x, y = np.meshgrid(np.arange(0, nx), np.arange(0, nx))
    colors = {}
    i = 0
    if interrogators:
        for interrogator in interrogators:
            colors[interrogator] = list(COLORS.values())[i]
            i += 1
    if verbose:
        print("Generating", len(quiver_x), "images.")
    for i in tqdm(range(len(quiver_x))):
        if interrogators:
            fig, ax = plt.subplots(1, len(interrogators)+1, figsize=((len(interrogators)+1)*5, 5), gridspec_kw={'width_ratios': [1 for _ in range(len(interrogators)+1)]})
            if c != []:
                ax[0].imshow(c.data, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
                ax[0].quiver(x, y, quiver_x[i]/nu_x, -quiver_y[i]/nu_y, scale=.25, units='xy')
            else:
                ax[0].quiver(x, y, quiver_x[i]/nu_x, quiver_y[i]/nu_y, scale=.25, units='xy')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            if c != []:
                ax.imshow(c.data, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
                ax.quiver(x, y, quiver_x[i]/nu_x, -quiver_y[i]/nu_y, scale=.25, units='xy')
            else:
                ax.quiver(x, y, quiver_x[i]/nu_x, quiver_y[i]/nu_y, scale=.25, units='xy')
        if interrogators:
            for inter in range(len(interrogators)):
                ax[0].scatter(interrogators[inter][0]+(nx//2), interrogators[inter][1]+(nx//2), marker="1", color=colors[interrogators[inter]])
                for j in range(len(interrogators_data[interrogators[inter]])):
                    ax[inter+1].plot(np.arange(0, len(quiver_x)*dt, dt)[:i+1], interrogators_data[interrogators[inter]][j][:i+1], linestyle=['-', '--', '-.'][j], color=colors[interrogators[inter]])
                ax[inter+1].set_xlabel("Time")
                ax[inter+1].set_ylabel("Amplitude")
                ax[inter+1].set_title(str(interrogators[inter]))
                ax[inter+1].set_ylim((np.min(np.array(list(interrogators_data.values()))), np.max(np.array(list(interrogators_data.values())))))
            ax[0].axis('off')
        fig.suptitle("t = " + str(dt*i)[:4] + "s")
        plt.tight_layout()
        plt.savefig(name + "%02d.png" % i, dpi=250)
        plt.close()
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', str(int(1/dt)), '-i', name + '%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name + ".mp4", '-y'
    ])
    for file_name in glob(name+"*.png"):
        remove(file_name)
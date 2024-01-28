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

def generate_video(img, interrogators, interrogators_data, name, nx, dt, c=None, verbose=False):
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
    for interrogator in interrogators:
        colors[interrogator] = list(COLORS.values())[i]
        i += 1
    if verbose:
        print("Generating", len(img), "images.")
    for i in tqdm(range(len(img))):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})
        if c != None:
            ax[0].imshow(c.data.T, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
        im = ax[0].imshow(img[i].T, cmap=get_black_cmap(), vmin=-np.max(np.abs(img[i:])), vmax=np.max(np.abs(img[i:])))
        plt.colorbar(im, shrink=0.75, ax=ax[0])
        for interrogator in interrogators:
            ax[0].scatter(interrogator[0]+(nx//2), -interrogator[1]+(nx//2), marker="1", color=colors[interrogator])
        for interrogator in interrogators:
            ax[1].plot(np.arange(0, len(img)*dt, dt)[:i+1], interrogators_data[interrogator][:i+1], color=colors[interrogator])
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Amplitude")
        ax[1].legend([str(i) for i in interrogators_data])
        ax[1].set_ylim((np.min(np.array(list(interrogators_data.values()))), np.max(np.array(list(interrogators_data.values())))))
        plt.title("t = " + str(dt*i)[:4] + "s")
        ax[0].axis('off')
        plt.savefig(name + "%02d.png" % i, dpi=250)
        plt.close()
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', str(int(1/dt)), '-i', name + '%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name + ".mp4", '-y'
    ])
    for file_name in glob(name+"*.png"):
        remove(file_name)
# PyAWD - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be

import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from subprocess import call
from os import remove, chdir
from tqdm.notebook import tqdm
from PyAWD.utils import *

def generate_video(img, name, dt, c=None, verbose=False):
    """
    Generates a video from a sequence of images.
    Arguments:
        - img: a list of 2d np.arrays representing the images
        - name: the name of the output file (without extension)
        - dt: the time interval between each images
        - c: the background image representing the velocity field
        - verbose: if True, displays logging informations
    """
    if verbose:
        print("Generating", len(img), "images.")
    for i in tqdm(range(len(img))):
        if c != None:
            plt.imshow(c.data, vmin=np.min(c.data), vmax=np.max(c.data), cmap="gray")
        plt.imshow(img[i], cmap=get_black_cmap(), vmin=-np.max(np.abs(img[i:])), vmax=np.max(np.abs(img[i:])))
        plt.title("t = " + str(dt*i)[:4] + "s")
        plt.colorbar(shrink=0.9) 
        plt.axis('off')
        plt.savefig(name + "%02d.png" % i, dpi=250)
        plt.close()
        
    call([
        'ffmpeg', '-loglevel', 'panic', '-framerate', str(int(1/dt)), '-i', name + '%02d.png', '-r', '32', '-pix_fmt', 'yuv420p',
         name + ".mp4", '-y'
    ])
    for file_name in glob("*.png"):
        remove(file_name)
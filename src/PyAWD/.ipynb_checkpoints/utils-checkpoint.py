# PyAWD - utils
# Tribel Pascal - pascal.tribel@ulb.be

from matplotlib.colors import LinearSegmentedColormap
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def get_white_cmap():
    """
    Creates a rose-white-green colormap
    """
    colors = [(1, 0, 0.7, 1), (1, 1, 1, 0.1), (0, 1, 0.7, 1)]
    return LinearSegmentedColormap.from_list('seism', colors)

def get_black_cmap():
    """
    Creates a rose-black-green colormap
    """
    colors = [(1, 0, 0.7, 1), (0, 0, 0, 0.1), (0, 1, 0.7, 1)]
    return LinearSegmentedColormap.from_list('seism', colors)

def get_ricker_wavelet(nx, A=0.1, x0=0, y0=0, sigma=0.075):
    """
    Generates a Ricker Wavelet
    Arguments:
        - nx: the grid size on which the wavelet is created
        - A: the scaling factor
        - x0: the center x coordinate (the grid is assumed to be centered in (0, 0))
        - y0: the center y coordinate (the grid is assumed to be centered in (0, 0))
        - sigma: the spreading factor
    Returns:
        - a np.array containing the generated Ricker Wavelet
    """
    x = np.arange(-1.-x0/(0.5*nx), 1.-x0/(0.5*nx), 2/nx)
    y = np.arange(-1.-y0/(0.5*nx), 1.-y0/(0.5*nx), 2/nx)
    x, y = np.meshgrid(x, y)
    return A * (2-x**2-y**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
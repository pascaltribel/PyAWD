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

def create_inverse_distance_matrix(nx, x0=0, y0=0, z0=0, tau=None, dim=2):
    """
    Creates an 1/distance matrix centered around (x0, y0)
    Arguments:
        - nx: the grid size on which the wavelet is created
        - x0: the center x coordinate (the grid is assumed to be centered in (0, 0))
        - y0: the center y coordinate (the grid is assumed to be centered in (0, 0))
        - tau: the distance threshold around (x0, y0) after which the distances are set to 0
        - dim: the number of dimensions of the generated field (2 or 3)
    Returns:
        - a np.array containing the generated explosive source
    """
    if not tau:
        tau = nx//2
    x = np.arange(nx)
    y = np.arange(nx)
    if dim==2:
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt((X-(x0+nx//2))**2 + (Y-(y0+nx//2))**2)
    elif dim==3:
        z = np.arange(nx)
        X, Y, Z = np.meshgrid(x, y, z)
        distance = np.sqrt((X-(x0+nx//2))**2 + (Y-(y0+nx//2))**2 + (Z-(z0+nx//2))**2)
    distance[distance > tau] = 0.
    distance[distance > 0] = 1/distance[distance > 0]
    return distance
    

def create_explosive_source(nx, x0=0, y0=0, z0=0, tau=None, dim=2):
    """
    Creates an explosive source (1/distance up to nx//10) centered around (x0, y0)
    Arguments:
        - nx: the grid size on which the wavelet is created
        - x0: the center x coordinate (the grid is assumed to be centered in (0, 0))
        - y0: the center y coordinate (the grid is assumed to be centered in (0, 0))
        - dim: the number of dimensions of the generated field (2 or 3)
    Returns:
        - a np.array containing the generated explosive source
    """
    if not tau:
        tau = nx//10
    if dim==2:
        s_x, s_y = create_inverse_distance_matrix(nx, x0, y0, tau=tau, dim=dim),\
                    create_inverse_distance_matrix(nx, x0, y0, tau=tau, dim=dim)
        s_x[:, :x0+nx//2] *= -1
        s_x[:, x0+nx//2] = 0
        
        s_y[:y0+nx//2] *= -1
        s_y[y0+nx//2] = 0
        
        res = s_x, s_y
        
    elif dim==3:
        s_x, s_y, s_z = create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim), \
                    create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim),\
                    create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim)
        s_x[:, :, :x0+nx//2] *= -1
        s_x[:, :, x0+nx//2] = 0
        
        s_y[:, :y0+nx//2] *= -1
        s_y[:, y0+nx//2] = 0

        s_z[:z0+nx//2] *= -1
        s_z[z0+nx//2] = 0

        res = s_x, s_y, s_z
        
    return res
# pyawd - utils
# Tribel Pascal - pascal.tribel@ulb.be
"""
Contains unclassable functions used across the package.
"""
import matplotlib.colors
import numpy as np


def get_white_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    """
    Returns:
         (matplotlib.colors.LinearSegmentedColormap): A rose-white-green colormap
    """
    colors = [(1, 0, 0.7, 1), (1, 1, 1, 0.1), (0, 1, 0.7, 1)]
    return matplotlib.colors.LinearSegmentedColormap.from_list('seismic_white', colors)


def get_black_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    """
    Returns:
         (matplotlib.colors.LinearSegmentedColormap): A rose-black-green colormap
    """
    colors = [(1, 0, 0.7, 1), (0, 0, 0, 0.1), (0, 1, 0.7, 1)]
    return matplotlib.colors.LinearSegmentedColormap.from_list('seismic_black', colors)


def get_ricker_wavelet(nx: int, a: float = 0.1, x0: int = 0, y0: int = 0, sigma: float = 0.075) -> np.meshgrid():
    """
    Generates a Ricker Wavelet
    Args:
        nx (int): The grid size on which the wavelet is created
        a (float): The scaling factor
        x0 (int): The center x coordinate (the grid is assumed to be centered in `(0, 0)`)
        y0 (int): The center y coordinate (the grid is assumed to be centered in `(0, 0)`)
        sigma (float): The spreading factor
    Returns:
        (numpy.meshgrid): A numpy meshgrid containing the generated Ricker Wavelet
    """
    x = np.arange(-1. - x0 / (0.5 * nx), 1. - x0 / (0.5 * nx), 2 / nx)
    y = np.arange(-1. - y0 / (0.5 * nx), 1. - y0 / (0.5 * nx), 2 / nx)
    x, y = np.meshgrid(x, y)
    return a * (2 - x ** 2 - y ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def create_inverse_distance_matrix(nx: int, x0: int = 0, y0: int = 0, z0: int = 0, tau: float = None, dim: int = 2) -> np.meshgrid:
    """
    Creates an $\\frac{1}{distance}$ matrix centered around `(x0, y0)`
    Args:
        nx (int): The grid size on which the wavelet is created
        x0 (int): The center x coordinate (the grid is assumed to be centered in `(0, 0)`)
        y0 (int): The center y coordinate (the grid is assumed to be centered in `(0, 0)`)
        z0 (int): The center z coordinate (the grid is assumed to be centered in `(0, 0, 0)`)
        tau (float): The distance threshold around (x0, y0) after which the distances are set to 0
        dim (int): The number of dimensions of the generated field (2 or 3)
    Returns:
        (numpy.meshgrid): A numpy meshgrid containing the generated explosive source
    """
    distance = np.array([])
    if not tau:
        tau = nx // 2
    x = np.arange(nx)
    y = np.arange(nx)
    if dim == 2:
        x, y = np.meshgrid(x, y)
        distance = np.sqrt((x - (x0 + nx // 2)) ** 2 + (y - (y0 + nx // 2)) ** 2)
    elif dim == 3:
        z = np.arange(nx)
        x, y, z = np.meshgrid(x, y, z)
        distance = np.sqrt((x - (x0 + nx // 2)) ** 2 + (y - (y0 + nx // 2)) ** 2 + (z - (z0 + nx // 2)) ** 2)
    distance[distance > tau] = 0.
    distance[distance > 0] = 1 / distance[distance > 0]
    return distance


def create_explosive_source(nx: int, x0: int = 0, y0: int = 0, z0: int = 0, tau: float = None, dim: int = 2) -> np.meshgrid:
    """
    Creates an explosive source ($\\frac{1}{distance}$ up to $\\lfloor\\frac{nx}{10}\\rfloor$) centered around `(x0, y0)`
    Args:
        nx (int): the grid size on which the wavelet is created
        x0 (int): the center x coordinate (the grid is assumed to be centered in `(0, 0)`)
        y0 (int): the center y coordinate (the grid is assumed to be centered in `(0, 0)`)
        z0 (int): the center z coordinate (the grid is assumed to be centered in `(0, 0)`)
        tau (float): the width of the explosive source
        dim (int): the number of dimensions of the generated field (2 or 3)
    Returns:
        (numpy.meshgrid): A numpy meshgrid containing the generated explosive source
    """
    res = np.array([])
    if not tau:
        tau = nx // 10
    if dim == 2:
        s_x, s_y = create_inverse_distance_matrix(nx, x0, y0, tau=tau, dim=dim), \
            create_inverse_distance_matrix(nx, x0, y0, tau=tau, dim=dim)
        s_x[:, :x0 + nx // 2] *= -1
        s_x[:, x0 + nx // 2] = 0

        s_y[:y0 + nx // 2] *= -1
        s_y[y0 + nx // 2] = 0

        res = s_x, s_y

    elif dim == 3:
        s_x, s_y, s_z = create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim), \
            create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim), \
            create_inverse_distance_matrix(nx, x0, y0, z0, tau, dim)
        s_x[:, :, :x0 + nx // 2] *= -1
        s_x[:, :, x0 + nx // 2] = 0

        s_y[:, :y0 + nx // 2] *= -1
        s_y[:, y0 + nx // 2] = 0

        s_z[:z0 + nx // 2] *= -1
        s_z[z0 + nx // 2] = 0

        res = s_x, s_y, s_z

    return res

# pyawd - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be
"""
Contains the Marmousi class.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyawd._marmousi_data import _get_marmousi_data


class Marmousi:
    """
    Represents the Marmousi velocity field. The maximal resolution is (955px*955px). This is only available in 2D.
    """
    nx: int = 32
    """
    The width of the field, in pixels
    """
    def __init__(self, nx: int = 32):
        """
        Args:
            nx (int): The width of the field, in pixels
        """
        self.raw_data = _get_marmousi_data()
        self.raw_nx = self.raw_data.shape[0]
        self.nx = min(nx, self.raw_nx)
        self.data = cv2.resize(self.raw_data, (nx, nx))
        self.data = self.data / (np.max(self.data) * 10)

    def get_data(self) -> np.ndarray:
        """
        Returns:
            - self.data: the velocity field
        """
        return self.data

    def plot(self):
        """
        Plots the field
        """
        plt.imshow(self.data)
        plt.show()

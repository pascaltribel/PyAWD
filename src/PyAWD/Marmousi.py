# PyAWD - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyAWD._marmousi_data import _get_marmousi_data

class Marmousi:
    """
    Represents the Marmousi velocity field. The current resolution is (955px*955px).
    Arguments:
        - nx: the discretisation of the array.
    """
    def __init__(self, nx):
        self.raw_data = _get_marmousi_data()
        self.raw_nx = self.raw_data.shape[0]
        self.nx = min(nx, self.raw_nx)
        self.data = cv2.resize(self.raw_data, (nx, nx))
        self.data = self.data/(np.max(self.data)*10)
    def get_data(self):
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
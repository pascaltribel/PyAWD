# PyAWD - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be

import numpy as np
import matplotlib.pyplot as plt
from PyAWD._marmousi_data import _get_marmousi_data

class Marmousi:
    """
    Represents the Marmousi velocity field. The current resolution is (955px*955px).
    Arguments:
        - nx: the discretisation of the array. If <955, it will lead to a subscaling to the field.
    """
    def __init__(self, nx):
        self.raw_data = _get_marmousi_data()
        self.raw_data = self.raw_data/(np.max(self.raw_data)*10)
        self.raw_nx = self.raw_data.shape[0]
        self.nx = min(nx, self.raw_nx)
        if nx != self.nx:
            print("The provided nx value is invalid. Using default value", self.nx)
        self.data = self.raw_data[::int(self.raw_nx/self.nx), ::int(self.raw_nx/self.nx)]

    def get_data(self):
        """
        Returns:
            - self.data: the subscaled velocity field
        """
        return self.data

    def plot(self):
        """
        Plots the field
        """
        plt.imshow(self.data)
        plt.show()
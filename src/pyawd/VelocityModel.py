# pyawd - VelocityModel
# Tribel Pascal - pascal.tribel@ulb.be
"""
Represents the velocity models as a numpy array
"""
import numpy as np


class VelocityModel:
    """
    Represents the velocity models as a numpy array
    """
    nx: int = 32
    """
    The width of the field, in pixels
    """
    dim: int = 3
    """
    The number of dimensions of the model
    """
    def __init__(self, nx: int = 32):
        """
        Args:
            nx (int): The width of the field, in pixels
        """
        self.nx = nx
        self.data = np.array([])

    def get_data(self) -> np.ndarray:
        """
        Returns:
            - self.data: the velocity field
        """
        return self.data

    def set_value(self, value: float):
        """
        Initialize the data array with the specified value
        Args:
            - value (float): the value to initialize the array with
        """
        self.data[:] = value

    def plot(self):
        """
        Plots the field
        """
        raise NotImplementedError('This class is abstract')

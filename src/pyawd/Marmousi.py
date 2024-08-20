# pyawd - Marmousi
# Tribel Pascal - pascal.tribel@ulb.be
"""
Contains the Marmousi class.
"""
import cv2
import numpy as np
from pyawd import VelocityModel2D
from pyawd._marmousi_data import _get_marmousi_data


class Marmousi(VelocityModel2D):
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
        super().__init__(min(nx, self.raw_nx))
        self.data = cv2.resize(self.raw_data, (nx, nx))
        self.data = self.data / (np.max(self.data) * 10)

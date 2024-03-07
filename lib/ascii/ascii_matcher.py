import cv2 as cv
import numpy as np


class AsciiMatcher:

    def __init__(self, characters=None):
        self.img = None
        self.characters = characters
        self.last_mask = None

    def get_pixel_diff(self, img1, img2):
        return np.count_nonzero(img1 - img2)

    def get_mse(self, img1, img2):

        """ Calculate the mean standard deviation of the difference between images. The lower the result, the smaller
        the difference. If the images are the same, the result is zero."""

        img1 = img1.astype(np.float64) / 255.
        img2 = img2.astype(np.float64) / 255.

        mse = np.mean(abs(img1 - img2) ** 2)

        return mse
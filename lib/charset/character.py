import functools

import numpy as np
import cv2 as cv
from cvtools import size_tools as tools

class Character:
    """
    Represents a 2 bit character (1s and 0s)
    """

    def __init__(self, img, index=None):
        self.img = img

        # Save downsampled pixel grayscale versions of the character img, for use in candidate selection
        self.grayscale = {}
        self.grayscale[1] = tools.resize_grayscale(img, (1,1))
        self.grayscale[2] = tools.resize_grayscale(img, (2,2))
        self.grayscale[4] = tools.resize_grayscale(img, (4,4))

        self.hash = None
        self.is_inverted = False

        self.height, self.width = img.shape[0], img.shape[1]

        # Reference to the hi-res version(s) of this character, only used by lo-res characters
        # A lo-res character can have multiple matching hi-res versions, hence we use a list
        self.high_res = set()

        # For this character, there can only be one low-res char per resolution, therefore we use
        # a map
        self.low_res = {}
        self.match_diff = 0

        self.index = index # Represents the index of the image for this character within the charset

        # This tuple will be used for cached methods which cannot take numpy arrays as arguments
        self.img_tuple = tuple(img.ravel().tolist())

    def is_empty(self):
        # Empty: all black pixels
        non_zero_pixels = np.count_nonzero(self.img)

        if non_zero_pixels == 0:
            return True
        else:
            return False

    def is_full(self):
        # Full: all white pixels
        non_zero_pixels = np.count_nonzero(self.img)

        if non_zero_pixels == self.width * self.height:
            return True
        else:
            return False

    def make_low_res(self, size):
        new_img = self.img

        # Adding dithering will reduce the number of possible outputs
        if size == 4:
            pass
            # new_img = self.add_dither(new_img)

        new_img = Character.img_resize(new_img, (size, size))

        low_res_char = Character(new_img)
        low_res_char.high_res.add(self)

        self.low_res[size] = low_res_char

        return low_res_char

    def add_dither(self, img):
        width, height = img.shape[1], img.shape[0]
        data = [[0, 255] * (width // 2), [255, 0] * (width // 2)] * (height // 2)

        pattern = np.asarray(data, dtype=np.uint8)
        img = cv.bitwise_and(img, pattern)

        return img

    @staticmethod
    def diff(source, target):
        """
        Calculates the number of pixels that differ between this character and another image of the same dimensions
        """
        max_diff = len(source) ** 2
        eq = source == target
        non_zero = np.count_nonzero(eq)
        return max_diff - non_zero

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def diff_cached(source: tuple, target: tuple, size):
        """
        Calculates the number of pixels that differ between this character and another image of the same dimensions
        """
        max_diff = size ** 2
        source = np.asarray(source, dtype=np.uint8)
        target = np.asarray(target, dtype=np.uint8)
        eq = source == target
        non_zero = np.count_nonzero(eq)
        return max_diff - non_zero

    def get_hash(self):
        if not self.hash:
            self.img.flags.writeable = False
            self.hash = hash(bytes(self.img))

        return self.hash

    @staticmethod
    def resize(img, size, threshold=64):
        """Utility function that automatically resizes both dimensions of an input image by a certain factor"""

        type = cv.INTER_CUBIC

        new_width = size[0]
        new_height = size[1]

        if new_width == 2:
            threshold = 128
        elif new_width == 4:
            threshold = 64

        img = cv.resize(img, (new_width, new_height), interpolation=type)

        # Threshold the results before returning (effectively making it 2-bit)
        # _, img = cv.threshold(img, threshold, 255, cv.ADAPTIVE_THRESH_MEAN_C)
        _, out_img = cv.threshold(img, threshold, 255, cv.THRESH_OTSU)

        # If it's blank, let's try again with a lower threshold
        if(new_width > 2):
            if np.count_nonzero(out_img) == 0:
                # Try again with a higher threshold
                _, out_img = cv.threshold(img, int(threshold * 0.5), 255, cv.THRESH_OTSU)
                pass
            elif np.count_nonzero(out_img) == (new_width * new_height):
                _, out_img = cv.threshold(img, int(threshold * 1.5), 255, cv.THRESH_OTSU)
                pass

        return out_img

    @staticmethod
    def img_resize(img, size):
        """Intended to be called to resize non character image blocks only """
        # 196 seems to be the best threshold, based on testing with "garcia-retrato"

        new_width = size[0]
        new_height = size[1]

        inter = cv.INTER_AREA
        threshold = new_width * 16

        if(new_width == 2):
            threshold = 128

        img = cv.resize(img, (new_width, new_height), interpolation=inter)
        _, out_img = cv.threshold(img, threshold, 255, cv.THRESH_OTSU)

        # If it's blank, let's try again with a higher threshold
        if np.count_nonzero(out_img) == 0:
            # Try again with a higher threshold
            _, out_img = cv.threshold(img, int(threshold * 1.5), 255, cv.THRESH_OTSU)
            pass
        elif np.count_nonzero(out_img) == (new_width * new_height):
            _, out_img = cv.threshold(img, int(threshold * 0.5), 255, cv.THRESH_OTSU)
            pass

        return out_img

    @staticmethod
    def pixel_resize(img, size):
        # return Character.resize(img, size, threshold=64)
        return cv.resize(img, (size[0], size[1]), interpolation=cv.INTER_NEAREST)

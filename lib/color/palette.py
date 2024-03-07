import functools
import os

import math

import cv2 as cv
import numpy as np

class Palette():
    '''Palette utilities to load and process 8-bit color palettes.

    The palette to load must be a color PNG composed of 8x8 pixel blocks of color.


    Credit to:
    @ref https://fornaxvoid.com/colorpalettes/

    For a lot of the included palette PNGs
    '''
    PALETTES_DIR = os.path.dirname(__file__) + "/res/palettes/"

    def __init__(self, colors=None, char_width = 8, char_height = 8):
        self.char_width = char_width
        self.char_height = char_height
        self.colors = []
        self.tree = None
        self.clusters = None # KMeans clusters

        if colors is not None:
            self.set_colors(colors)

    def load(self, filename):
        palette = cv.imread(self.PALETTES_DIR + filename, cv.IMREAD_UNCHANGED)

        colors = []

        width, height = palette.shape[0], palette.shape[1]

        # Slice up the palette into 8x8 blocks
        for x in range(0, width, self.char_width):
            for y in range(0, height, self.char_height):
                pixel = palette[x, y]
                colors.append(pixel)

        self.set_colors(colors, 256)


    def set_colors(self, colors, max=256):
        if len(colors) > max:
            colors = colors[0:max]

        self.colors = np.array(colors)


    """ DEPRECATED """
    @functools.lru_cache(maxsize=None)
    def find_closest_color(self, source_color):
        # _, match_idx = self.tree.query([source_color], 1)
        #
        # return match_idx[0], self.tree.data[match_idx[0]]
        labels = self.clusters.predict([source_color])

        return 0, labels


    def get_color_distance(self, color1, color2):
        '''Get the distance between two colors: the closer the colors, the smaller the distance

        @ref http://stackoverflow.com/questions/2103368/color-logic-algorithm

        Colors passed are simply 3 element arrays of uint8 type (8 bit), where color = [blue, green, red]
        In other words, colors are stored as the usual BGR format of OpenCV
        '''
        color1 = np.array(color1, np.uint8)
        color2 = np.array(color2, np.uint8)
        # color1_signed = color1.astype(np.int16, casting='same_kind')
        # color2_signed = color2.astype(np.int16, casting='same_kind')
        color1_signed = color1
        color2_signed = color2

        rmean = (color1_signed[2] + color2_signed[2]) / 2

        red = color1_signed[2] - color2_signed[2]
        green = color1_signed[1] - color2_signed[1]
        blue = color1_signed[0] - color2_signed[0]

        weight_red = 2 + (rmean / 256)
        weight_green = 4.0
        weight_blue = 2 + (255 - rmean) / 256
        # weight_red = weight_green = weight_blue = 1

        distance = math.sqrt((weight_red * red * red) + (weight_green * green * green) + (weight_blue * blue * blue))

        return distance


    def sort_colors(self):
        """
        Sort the palette by the norm of each pixel vector
        :return:
        """
        self.colors = sorted(self.colors, key=lambda x:cv.norm(x, normType=cv.NORM_L2SQR))
        return self.colors


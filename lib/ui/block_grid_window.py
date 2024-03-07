import os

import numpy as np

from ascii import AsciiConverter
from ui.opencv_window import Window
import cv2 as cv

class BlockGridWindow(Window):
    """An OpenCV window that allows you to select character blocks for picking ASCII Matches"""

    def __init__(self, window_name):
        super(BlockGridWindow, self).__init__(window_name)
        self.char_width = 8
        self.char_height = 8
        self.selected_block = None
        self.show_grid = False
        self.show_layer = None
        self.char_picker_window = None
        self.converter:AsciiConverter = None
        self.image_path = None

    def show(self, img):
        if self.selected_block:
            self.add_selected_outline(img)

        if self.show_grid:
            self.add_grid(img)

        self.img = img
        return super().show(img)

    def select_block(self, block_coords):
        self.selected_block = block_coords

    def add_selected_outline(self, img):

        char_width, char_height = self.char_width * 2, self.char_height * 2
        outline_color = (0, 255, 0)
        row, col = self.selected_block
        start_x, end_x = col * char_width, (col * char_width) + char_width
        start_y, end_y = row * char_height, (row * char_height) + char_height

        cv.rectangle(img, (start_x, start_y), (end_x, end_y), outline_color)

    def set_override_char(self, char_code):
        """ Keep track of a manually set overriden ASCII character in a 2D grid. This should be called from the
        character picker window"""
        row, col = self.selected_block
        self.converter.set_override_char(char_code, (row, col))


    def add_grid(self, img):
        grid_color = (0, 0, 255)
        height = img.shape[0]
        width = img.shape[1]

        for x in range(0, width, self.char_width * 2):
            cv.line(img, (x, 0), (x, height), grid_color)

        for y in range(0, height, self.char_height * 2):
            cv.line(img, (0, y), (width, y), grid_color)

    def get_key(self):
        """Arrow keys allow selection of a character from the candidate list, in order to manually
        override algorithm selected characters"""

        key = super().get_key()

        # These are most likely window specific key codes

        key_left = 2424832
        key_up = 2490368
        key_right = 2555904
        key_down = 2621440

        if self.char_picker_window:
            if key == key_left:
                self.char_picker_window.key_left()
            elif key == key_right:
                self.char_picker_window.key_right()
            elif key == key_up:
                self.char_picker_window.key_up()
            elif key == key_down:
                self.char_picker_window.key_down()
            elif key == ord('v'):
                # Saves the ASCII override data to a file
                path, file = os.path.split(self.image_path)

                filename = os.path.join(path, (file + '.txt'))
                with open(filename, 'w') as file:
                    file.write(self.converter.get_override_csv_data())

                print(f'Character data file saved to: {filename}')


        return key

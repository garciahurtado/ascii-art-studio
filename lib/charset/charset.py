import json
import math
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import unicodedata

from debugger import printc
from .character import Character


class Charset:
    """Charset utilities to load and process 8-bit style character sets.

    The charsets to be loaded must include all characters in a single black and white PNG image.
    Each character should be 8x8 pixels. The number of rows and columns is not important.
    """

    CHARSETS_DIR = os.path.dirname(__file__) + "/res/charsets/"

    def __init__(self, char_width=8, char_height=8):
        self.char_width = char_width
        self.char_height = char_height
        self.chars = []
        self.pixel_histogram = None
        self.charset_img = None
        self.filename = None
        self.hex_codes = None
        self.inverted_included = False

        self.full_char = Character(np.full((char_height, char_width), 255, dtype=np.uint8))
        full_med = self.full_char.make_low_res(4)
        full_med.make_low_res(2)

        self.empty_char = Character(np.zeros((char_height, char_width), dtype=np.uint8))
        empty_med = self.empty_char.make_low_res(4)
        empty_med.make_low_res(2)


    def __iter__(self):
        """ Make iterable """
        return iter(self.chars)

    def __len__(self):
        """ Make it work with len"""
        return len(self.chars)

    def load(self, filename, invert=False):
        """
        Load a charset from a black and white PNG image on disk and slice it into individual characters.
        If specified, also create inverted versions of all the loaded characters.
        """
        self.filename = filename
        filename = self.CHARSETS_DIR + filename

        if not os.path.exists(filename):
            raise FileNotFoundError(f'Unable to load charset. File "{filename}" does not exist')

        charset = cv.imread(filename)
        charset = cv.cvtColor(charset, cv.COLOR_BGR2GRAY)
        self.charset_img = charset

        charset_width = charset.shape[1]
        charset_height = charset.shape[0]
        print(f"Loaded charset of size: {charset_width}x{charset_height} / character size: {self.char_width}x{self.char_height}")

        self.load_metadata(filename)
        if self.hex_codes is not None:
            print(f"length: {len(self.hex_codes)}")

        # Slice up the charset image into individual character blocks
        for y in range(0, charset_height, self.char_height):
            for x in range(0, charset_width, self.char_width):
                img = charset[y:y + self.char_height, x:x + self.char_width]
                idx = len(self.chars)
                hex_code = self.hex_codes[idx] if self.hex_codes and idx < len(self.hex_codes) else None
                new_char = Character(img, idx, hex_code)

                # First character made of all white pixels, keep track of it, so we dont end up with duplicates
                if new_char.is_full():
                    if self.full_char.index is None:
                        self.full_char.index = idx
                        self.full_char.code = new_char.code
                        self.chars.append(new_char)
                elif new_char.is_empty(): # First character made of all black pixels, keep track of it
                    if self.empty_char.index is None:
                        self.empty_char.index = idx
                        self.empty_char.code = new_char.code
                        self.chars.append(new_char)
                else:
                    self.chars.append(new_char)

        if invert:
            inverted_char = None
            inv_chars = []
            idx = 0

            for character in self.chars:
                idx = len(self.chars) + len(inv_chars)
                inverted_char = Character(cv.bitwise_not(character.img), idx)
                inverted_char.is_inverted = True

                # First character made of all black pixels, keep track of it
                if inverted_char.is_full():
                    if self.full_char.index is None:
                        self.full_char.index = idx
                        inv_chars.append(inverted_char)
                elif inverted_char.is_empty():
                    if self.empty_char.index is None:
                        self.empty_char.index = idx
                        inv_chars.append(inverted_char)
                else:
                    inv_chars.append(inverted_char)

            self.chars.extend(inv_chars)


        print(f"{len(self.chars)} total characters loaded")


        return

    def load_metadata(self, img_file):
        # Remove .png from the img filename and add .json
        json_file = img_file.replace(".png", ".json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                all_json = json.load(file)
                if "inverted_included" in all_json.keys():
                    self.inverted_included = all_json["inverted_included"]

                if "hex_codes" in all_json.keys():
                    self.hex_codes = all_json["hex_codes"]
                    if self.inverted_included:
                        self.hex_codes = self.hex_codes + self.hex_codes
        else:
            printc(f"!! No metadata file {json_file} found !!")

    def print_unicode_chars(self, show_hex=False):
        if not self.hex_codes:
            printc("!! No hex codes available for this charset !!")
            return

        total = 0

        for i, hex_code in enumerate(self.hex_codes, 1):
            try:
                character = chr(int(hex_code, 16))

                character = character + hex_code if show_hex else character
                print(f"{character}", end="")

                total = total + 1

                if total % 16 == 0:
                    print()
            except ValueError:
                printc(f"Invalid hex code: {hex_code}", end="\t")

                if i % 16 == 0:
                    print()

        if len(self.hex_codes) % 16 != 0:
            print()

        print(f"{total} characters printed")

    def write(self, filename):
        """
        Write this charset to disk as a black and white PNG, trying to arrange the total number of characters
        in an even number of rows and columns
        """
        chars = self.chars.copy()

        num_chars = len(chars)
        num_cols = math.floor(math.sqrt(num_chars))
        num_rows = num_cols
        extra_chars = num_chars - (num_cols * num_rows)
        num_pad_chars = 0

        if extra_chars > 0:
            print(f"Extra chars: {extra_chars}")
            # Can't fit in a square, add one or two more rows
            num_rows += math.ceil(extra_chars / num_cols)
            num_pad_chars = num_cols - (extra_chars % num_cols)

        rows = []

        # TODO: Needs to be updated, only does single column right now
        for i in range(0, num_rows):

            # We may need to pad the last row if it's a jagged array
            if (i == num_rows - 1) and (num_pad_chars > 0):
                print(f"Num pad chars: {num_pad_chars}")
                empty_char = self.empty_char
                padding = [empty_char for i in range(num_pad_chars)]
                chars.extend(padding)

            all_row_chars = []

            for j in range(0, num_cols):
                idx = (i*num_cols) + j
                all_row_chars.append(chars[idx].img)

            rows.append(cv.hconcat(all_row_chars))

        charmap = cv.vconcat(rows)
        cv.imwrite(self.CHARSETS_DIR + filename, charmap)

    def make_charset_index_table(charset):
        """Generate an image showing each character and their corresponding index in the charset they were loaded from.
        This is needed for model training, as those indexes will become the labels"""
        char_width, char_height = charset.char_width, charset.char_height
        size = (len(charset) * char_height, char_width * 5)
        img = np.zeros(size, dtype=np.uint8)

        for i, character in enumerate(charset):
            img[i * char_height: (i + 1) * char_height, 0:char_width] = character.img
            color = (255, 255, 255)
            cv.putText(img, str(character.index), (char_width + 3, ((i + 1) * char_height) - 1), cv.FONT_HERSHEY_PLAIN,
                       .75, color)

        return img

    def make_charset_sprite(charset):
        '''Generate a single sprite containing the images of every character in the charset, to be used as image labels for
        Tensorboard'''

        char_width, char_height = 8, 16
        cols = math.ceil(math.sqrt(len(charset)))
        size = (cols * char_height, cols * char_width)
        img = np.zeros(size, dtype=np.uint8)

        for i, character in enumerate(charset):
            col = i % cols
            row = math.floor(i / cols)

            img[row * char_height: (row + 1) * char_height, col * char_width: (col + 1) * char_width] = character.img

        return img

    def show_histogram(self):
        weights = {}
        for char in self.chars:
            char.flatten()
            char_weight = char.sum()
            if char_weight not in weights.keys():
                weights[char_weight] = 1
            else:
                weights[char_weight] += 1

        lists = sorted(weights.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.show()

    def show_low_res_maps(self, grid=True):
        """Returns an image showing the low-res maps and their associated full-res characters"""

        char_width = char_box_width = self.char_width
        char_height = char_box_height = self.char_height

        if grid:
            char_box_width += 1
            char_box_height += 1

        img_width = char_box_width * 3
        img_height = len(self.chars) * char_box_height * 2

        # Initialize a new black and white image with black background
        full_map = np.zeros((img_height, img_width), np.uint8)
        num_cols = 12

        character_2: Character
        character_4: Character
        character_8: Character

        i = 0
        for character_8 in self.chars:

            character_4 = character_8.low_res[4]
            character_4_img = Character.pixel_resize(character_4.img, (8, 8))

            character_2 = character_4.low_res[2]
            character_2_img = Character.pixel_resize(character_2.img, (8, 8))

            y = i * char_box_height + 1
            x_1 = 1
            x_2 = char_box_width + 1
            x_3 = char_box_width * 2 + 1

            full_map[y : y + char_height, x_1 : x_1 + char_width] = character_8.img
            full_map[y : y + char_height, x_2 : x_2 + char_width] = character_4_img
            full_map[y : y + char_height, x_3 : x_3 + char_width] = character_2_img

            i += 1

        if num_cols > 1:
            columns = []
            col_height = math.ceil(img_height / num_cols)

            for col_num in range(0, num_cols):
                y = col_num * col_height
                columns.append(full_map[y : y + col_height, 0 : char_box_width * 3])

            full_map = np.hstack(columns)

        full_map = cv.cvtColor(full_map, cv.COLOR_GRAY2RGB)
        img_height, img_width = full_map.shape[0], full_map.shape[1]

        if grid:
            grid_color = (0, 92, 0)
            col_color = (0, 255, 0)

            # Horizontal lines
            for y in range(0, img_height, char_box_height):
                cv.line(full_map, (0, y), (img_width, y), grid_color, 1)

            # Vertical lines
            for x in range(0, img_width, char_box_width):
                cv.line(
                    full_map, (x, 0), (x, char_box_height * img_height), grid_color, 1
                )

            # Column separators
            for x in range(0, img_width, char_box_width * 3):
                cv.line(
                    full_map, (x, 0), (x, char_box_height * img_height), col_color, 1
                )

        full_map = Character.pixel_resize(full_map, (img_width * 2, img_height * 2))
        return full_map

    def nested_dict(self, n, type):
        """@ref https://stackoverflow.com/questions/29348345/declaring-a-multi-dimensional-dictionary-in-python"""
        if n == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: self.nested_dict(n - 1, type))

    def get_pixel_histogram(self):
        """ Generates a 8x8 image representing the frequency of white pixels averaged across all characters in this set"""

        img_sum = np.zeros([self.char_width, self.char_height], dtype=np.float)

        for char in self.chars:
            img_sum = img_sum + char.img

        # Calculate average value of each pixel
        num_chars = len(self.chars)
        img_sum = img_sum / num_chars
        img_sum = img_sum.astype(np.uint8)

        return img_sum

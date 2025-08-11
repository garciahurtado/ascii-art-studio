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
    this_dir = os.path.dirname(__file__)
    CHARSETS_DIR = os.path.join(this_dir, "res", "charsets")

    def __init__(self, char_width=8, char_height=8):
        self.char_width = char_width
        self.char_height = char_height
        self.chars = []
        self.num_chars = 0
        self.skip_chars = []
        self.pixel_histogram = None
        self.charset_img = None
        self.filename = None
        self.name = None
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
        self.name = self.filename.replace(".png", "")
        filename = os.path.join(self.CHARSETS_DIR, filename)

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
                if idx >= self.num_chars:
                    break

                hex_code = self.hex_codes[idx] if self.hex_codes and idx < len(self.hex_codes) else None
                new_char = Character(img, idx, hex_code)

                # Any character that is empty or full should be ignored, since they already exist in the charset object
                if new_char.is_full() and not self.full_char.index:
                    printc(f"Found full char with idx:{new_char.index}")

                elif new_char.is_empty() and not self.empty_char.index:  # First character made of all black pixels, keep track of it
                    printc(f"Found empty char with idx:{new_char.index}")
                else:
                    self.chars.append(new_char)

        if invert:
            inv_chars = []

            for character in self.chars:
                idx_inv = idx + len(inv_chars)
                inverted_char = Character(cv.bitwise_not(character.img), idx_inv)
                inverted_char.is_inverted = True

                # ignore all empty and full characters, so we dont end up with duplicates. We will add them later
                if inverted_char.is_full() or inverted_char.is_empty():
                    continue
                else:
                    inv_chars.append(inverted_char)

            self.chars.extend(inv_chars)

        # we dont want to include the full and empty characters in the character count
        self.num_chars = len(self.chars)

        # Add the full and empty characters
        self.full_char.index = len(self.chars)
        self.chars.append(self.full_char)

        self.empty_char.index = len(self.chars)
        self.chars.append(self.empty_char)

        print(f"{self.num_chars} total characters loaded from charset, +2 more for full and empty")

        return

    def load_metadata(self, img_file):
        # Remove .png from the img filename and add .json
        json_file = img_file.replace(".png", ".json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                all_json = json.load(file)
                self.num_chars = all_json["num_chars"]
                self.char_width = all_json["char_width"]
                self.char_height = all_json["char_height"]

                if "inverted_included" in all_json.keys():
                    self.inverted_included = all_json["inverted_included"]

                if "hex_codes" in all_json.keys():
                    self.hex_codes = all_json["hex_codes"]
                    if self.inverted_included:
                        self.hex_codes = self.hex_codes + self.hex_codes

                if "skip_chars" in all_json.keys():
                    self.skip_chars = all_json["skip_chars"]
        else:
            printc(f"!! No metadata file {json_file} found !!")

    def save_metadata(self):
        # Remove .png from the img filename and add .json
        json_file = os.path.join(self.CHARSETS_DIR, self.name + ".json")

        with open(json_file, 'w') as file:
            conf = {
                "name": self.name,
                "char_height": self.char_height,
                "char_width": self.char_width,
                "num_chars": len(self.chars),
                "inverted_included": self.inverted_included
            }
            if self.hex_codes:
                conf["hex_codes"] = self.hex_codes
            if self.skip_chars:
                conf["skip_chars"] = self.skip_chars

            json.dump(conf, file, indent=4)
            print(f"Saved charset metadata to {json_file}")

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

    def write(self, filename, skip_chars=None):
        """
        Write this charset to disk as a black and white PNG, trying to arrange the total number of characters
        in an even number of rows and columns
        """
        orig_chars = self.chars.copy()
        print(f"Number of original characters: {len(orig_chars)}")
        num_skipped = 0

        if skip_chars:
            new_chars = []

            # Remove the list of characters to skip from the original
            for char in orig_chars:
                empty_or_full = char.is_empty() or char.is_full()
                if (int(char.index) not in skip_chars) and not empty_or_full:
                    new_chars.append(char)
                else:
                    num_skipped = num_skipped + 1
        else:
            new_chars = orig_chars

        self.chars = new_chars

        num_chars = len(new_chars)
        write_chars = new_chars.copy()
        num_rows = 16
        num_cols = math.ceil(num_chars / num_rows)
        num_pad_chars = num_cols - (num_chars % num_cols)

        rows = []

        for i in range(0, num_rows):

            # We may need to pad the last row if it's a jagged array
            if (i == num_rows - 1) and (num_pad_chars > 0):
                empty_char = self.empty_char
                padding = [empty_char for i in range(num_pad_chars)]
                write_chars.extend(padding)

            all_row_chars = []

            for j in range(0, num_cols):
                idx = (i*num_cols) + j
                all_row_chars.append(write_chars[idx].img)

            rows.append(cv.hconcat(all_row_chars))

        charmap = cv.vconcat(rows)
        final_path = os.path.join(self.CHARSETS_DIR, filename)
        cv.imwrite(final_path, charmap)

        if not num_skipped: num_skipped = "none"
        print(f"Charset image with {num_chars} chars ({num_skipped} skipped) written to: {final_path}")

    def write_without_skip_chars(self):
        """
        If the metadata file has a list of skip_chars, remove them from the charset before packing and writing to disk
        """
        # remove the extension from the filename

        if self.skip_chars:
            old_skip_chars = self.skip_chars

            # Save new charmap image and metadata file
            self.name = self.name + "-lean"
            filename = self.name
            self.write(filename + ".png", self.skip_chars)

            # We've already saved the new charset image without skip_chars, so remove them from the metadata
            self.skip_chars = False
            self.save_metadata()
            self.skip_chars = old_skip_chars
        else:
            raise Exception("No skip_chars present in metadata json")

    def make_charset_index_table(self):
        """Generate an image showing each character and their corresponding index in the charset they were loaded from.
        This is needed for model training, as those indexes will become the labels"""
        char_width, char_height = self.char_width, self.char_height
        size = (len(self) * char_height, char_width * 5)
        img = np.zeros(size, dtype=np.uint8)

        for i, character in enumerate(self.chars):
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

import cachetools
import math
from cachetools import cached, LRUCache

from charset import Character
from charset import Charset
import cv2 as cv
import numpy as np
import cvtools.contrast_filters as filters

"""Uses a Charset object in order to convert a given image into an ASCII version manually (ie: no
machine learning)

These converted images can then be used for training a machine learning model
"""

class AsciiConverter:

    diff_empty_threshold = 1
    diff_full_threshold = 63
    diff_match_min_threshold = 0
    diff_match_max_threshold = 64

    def __init__(self, charset:Charset, resolutions=(2, 4, 8)):
        self.charset:Charset = charset
        self.char_width = charset.char_width
        self.char_height = charset.char_height
        self.root_chars = set()
        self.used_chars = []
        self.match_char_map = None
        self.candidate_chars = None
        self.img_block_map = None
        self.mask_map = None
        self.region = False # Only convert this region of the image, for performance and testing
        self.last_mask = None
        self.override_chars = None # 2D list of manual override characters, indexed by column and row


        # List of resolutions to be used in character conversion in order to speed up image matching
        self.resolutions = resolutions

        self.create_all_low_res()


    def set_region(self, start:tuple, end:tuple):
        """Limit the ASCII conversion to the region determined by the start (x,y) and end (x,y) arguments"""
        self.region = [list(value) for value in zip(start, end)]


    def set_override_char(self, char_code, coords):
        """ Keep track of a manually set overriden ASCII character in a 2D grid. This should be called from the
        character picker window"""
        row, col = coords
        character = self.charset.chars[char_code] # The char code matches the index, since they are loaded in order

        self.override_chars[row, col] = char_code


    def convert_image(self, input_image):
        width, height = input_image.shape[1], input_image.shape[0]
        block_cols = math.ceil(width / self.char_width)
        block_rows = math.floor(height / self.char_height)

        # initialize the grid of manually overriden ASCII characters
        if self.override_chars is None:
            rows, cols = int(input_image.shape[0] / self.char_height), int(input_image.shape[1] / self.char_width),
            self.override_chars = np.full((rows, cols), -1, dtype=np.int16)

        # Slice up the B&W input image into blocks and match each of them to an ASCII characters

        self.used_chars = [] # Keep track of the unique characters used in this rendering, for analytics
        self.match_char_map = np.full((block_rows, block_cols), None, dtype=object)
        self.candidate_chars =  [[[] for i in range(block_cols)] for j in range(block_rows)]
        self.img_block_map =  np.full((block_rows, block_cols), None, dtype=object)
        self.mask_map = np.full((block_rows, block_cols), None, dtype=object)

        output_image = input_image.copy()

        for y in range(0, height, self.char_height):
            row = int(y / self.char_width)

            for x in range(0, width, self.char_width):
                col = int(x / self.char_height)

                # Check whether block is inside the region to be converted
                if(self.region):
                    if (col not in range(*self.region[0]) ) or \
                            (row not in range(*self.region[1]) ):

                        continue

                img_block = input_image[y:y + self.char_height, x:x + self.char_height]
                self.img_block_map[row][col] = img_block

                matches = self.find_char_matches(img_block)

                # Keep some character and image data about this conversion
                self.candidate_chars[row][col] = matches

                match = self.get_best_match(matches, img_block)

                # Apply override character, if available
                char = self.get_override_character((row, col))
                if char:
                    match = char

                self.match_char_map[row][col] = match

                if(self.last_mask is not None):
                    self.mask_map[row][col] = self.last_mask

                self.used_chars.append(match)

                match_img = match.img

                output_image[y:y + self.char_height, x:x + self.char_width] = match_img

        return output_image


    def create_all_low_res(self):
        """Create lower resolution versions of the existing characters.
        Original: 8x8
        Low res: 4x4, 2x2
        """
        character: Character

        low_res_chars = self.charset.chars

        for res in reversed(self.resolutions):
            if res == self.char_width:
                low_res_chars = self.charset.chars
            else:
                low_res_chars = self.create_lower_res(low_res_chars, res)

        self.root_chars = low_res_chars

        return


    def create_lower_res(self, hires_chars, size):
        """
        Given a list of characters, create a lower resolution version of each, ensuring no duplicates
        In order to avoid duplicates, we get the hash from each generated character (which is a unique
        representation of the pixels it contains) and store it in a map as we go.

        :param hires_chars: List of characters to convert
        :param size: The size to convert to (in pixels)
        :return: List of low res characters generated
        """

        low_res_chars = {}

        for character in hires_chars:
            # Create lower character of specified size
            low_res: Character = character.make_low_res(size)

            hash = low_res.get_hash()

            if hash in low_res_chars.keys():
                low_res = low_res_chars[hash]
                low_res.high_res.add(character)
            else:
                low_res_chars[hash] = low_res

            character.low_res[size] = low_res

        return low_res_chars.values()


    def find_char_matches(self, img):
        """
        Given an image block, iterate through all the characters in this set until one is found that best
        resembles the pixels in the passed image
        """

        # If the chunk is nearly all 1s, or nearly all 0s (depending on threshold),
        # return a solid block of 1s or 0s and bail early
        if np.count_nonzero(img) > self.diff_full_threshold:
            return [self.charset.full_char]
        elif np.count_nonzero(img) < self.diff_empty_threshold:  # This actually counts zeros
            return [self.charset.empty_char]

        next_res_chars = self.root_chars

        for res in self.resolutions:

            if len(next_res_chars) > 1:
                if(res < 8):
                    img_low_res = Character.img_resize(img, (res, res))
                else:
                    img_low_res = img

                perfect_matches = self.find_perfect_matches(next_res_chars, img_low_res)
                close_matches = self.find_closest_matches(next_res_chars, img_low_res, res)

                perfect_matches.extend(close_matches)
                next_res_chars = perfect_matches

            # There's gotta be a more pythonic way of extracting the high_res array from each character
            if(res < 8):
                _new = []
                for match in next_res_chars:
                    for high_res_char in match.high_res:
                        high_res_char.match_diff = match.match_diff
                        _new.append(high_res_char)

                next_res_chars = _new

        num = len(next_res_chars)
        return next_res_chars


    def find_perfect_matches(self, characters, img):
        """
        Given a list of ASCII characters (at any resolution), and an image, find and return
        only the character(s) which match the pixels of the image perfectly
        """
        perfect_matches = []
        size = img.shape[0]

        if size <= 8:
            img_tuple = tuple(img.flatten().tolist())
        else:
            img_tuple = None

        for possible_match in characters:

            # Rather than counting pixels, use the hash of each character
            # if possible_match.get_hash() == img_char.get_hash():  # perfect match
            possible_match.match_diff = 0

            if img_tuple:
                this_diff = Character.diff_cached(possible_match.img_tuple, img_tuple, size)
            else:
                this_diff = Character.diff(possible_match.img, img)

            if(this_diff == 0):
                perfect_matches.append(possible_match)

        return perfect_matches


    def find_closest_matches(self, characters, img, size):
        """
        Given a list of ASCII characters, and an image, find and return the character(s) which best match the pixels
        in the image
        """
        prev_diff = size * size + 1
        close_matches = []

        fuzz_factor = self.calculate_fuzz_factor(size)

        if size <= 8:
            img_tuple = tuple(img.flatten().tolist())
        else:
            img_tuple = None

        for possible_match in characters:
            # Count the number of pixels which are different between the image chunk and the current character

            if(img_tuple):
                this_diff = Character.diff_cached(possible_match.img_tuple, img_tuple, size)
            else:
                this_diff = Character.diff(possible_match.img, img)

            possible_match.match_diff = this_diff
            next_lower  = int(size/2)

            # Accumulate diff from parent low res character, but only if the diff is more than the fuzz factor
            if((this_diff > fuzz_factor) and (next_lower in possible_match.low_res)):
                if(size == 8):
                    # We add it a second time, since errors at the highest res are the most significant
                    possible_match.match_diff += this_diff / 2
                else:
                    possible_match.match_diff += possible_match.low_res[next_lower].match_diff * (size / 4)

            diff_delta = this_diff - prev_diff

            if diff_delta > fuzz_factor:
                # The new match is worse, forget it
                continue

            if diff_delta < - fuzz_factor:
                # We found a better match than the last one, so forget about the last one and keep track of it
                close_matches = []
                close_matches.append(possible_match)
                prev_diff = this_diff

            elif abs(diff_delta) <= fuzz_factor:
                # Both this match and the previous one are about as good
                close_matches.append(possible_match)

        return close_matches


    def calculate_fuzz_factor(self, size):
        fuzz_factor = 0

        if size < 2:
            fuzz_factor = 0
        elif size == 2:
            fuzz_factor = 1
        elif size == 3:
            fuzz_factor = 4
        elif size == 4:
            fuzz_factor = 5
        elif size >= 8:
            fuzz_factor = 0

        return fuzz_factor

    def get_best_match(self, charlist, img=None):
        """ Given a list of already matched characters, return the one with the lowest diff score """
        final_match = charlist[0]

        if (len(charlist) == 1):
            return charlist[0]
        else:
            lowest_diff = self.char_width * self.char_width

            for best_match in charlist:
                if best_match.match_diff <= lowest_diff:
                    final_match = best_match
                    lowest_diff = best_match.match_diff

        return final_match

    def count_used_chars(self):
        """Returns the unique number of characters used during the last image conversion"""

        return len(set(self.used_chars))


    def get_pixel_diff(self, img1, img2):
        return np.count_nonzero(img1 - img2)


    def get_csv_data(self):
        """Collect conversion data from a specific ASCII converted image and return it in a text format that
        can be used for training an ML model"""

        characters = self.match_char_map
        img_blocks = self.img_block_map

        all_blocks = []

        for y, row in enumerate(characters):
            for x, _ in enumerate(row):
                character = characters[y][x]
                img_block = img_blocks[y][x]

                item = f"{character.index}\t" + filters.get_as_csv(img_block)
                all_blocks.append(item)

        return "\n".join(all_blocks)

    def get_label_data(self):
        """Collect conversion data from a specific ASCII converted image, labels only. Returned as an ordered list"""

        labels = [char.index for char in self.used_chars]
        return labels


    def get_override_csv_data(self):
        """Collect ASCII characters manually overriden and return them as a CSV string """

        characters = self.override_chars

        all_rows = []

        for y, row in enumerate(characters):
            row_data = "\t".join(str(code) for code in row)
            all_rows.append(row_data)

        return "\n".join(all_rows)


    def get_override_character(self, coords):
        row, col = coords

        if self.override_chars is not None:
            code = self.override_chars[row, col]
            if code != -1:
                character = self.charset.chars[code]
                return character

        return False



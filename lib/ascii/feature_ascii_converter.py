import itertools
import math
from typing import Callable

import numpy as np
from skimage import io, transform, feature, exposure, util
from scipy.spatial import distance

from charset import Charset, Character
from ascii.shifting_ascii_converter import ShiftingAsciiConverter
import cv2 as cv


class FeatureAsciiConverter(ShiftingAsciiConverter):
    """ASCII Converter that uses the HOG (histogram of oriented gradients) algorithm to calculate image differences"""

    def __init__(self, charset:Charset):
        super(FeatureAsciiConverter, self).__init__(charset)

        self.diff_full_threshold = 62
        self.diff_empty_threshold = 3
        # number of candidate matches to return on first pass, decrease to improve overall performance
        self.num_close_matches = 256 # 256 is best fidelity
        self.pixel_weights = None


        self.match_char_map = None
        self.dist_eucl_map = None
        self.pixel_diff_map = None
        self.pixel_factor_map = None
        self.hog_factor_map = None


        # To make selection of candidates more efficient
        self.white_pixel_indices = {}

        # Map where we will cache the HOGs of each character in the set
        self.char_feat_map = {}
        self.calculate_character_features()

        self.shift_dims = (1, 0, -1)

        # List of pixel shifted character images, to speed up diff calculations
        self.shifted_chars = {}
        self.generate_shifted_chars()

        # Characters sorted by number of white pixels, for fast finding of match candidates
        self.pixel_sorted_chars = self.sort_by_white_pixels(self.charset)


    def sort_by_white_pixels(self, charset):
        out_list = sorted(charset.chars, key=lambda char: np.count_nonzero(char.img))

        # Generate indices of the middle character in a group of same pixel count characters
        for pixel_count in range(0,self.char_width * self.char_height):
            start_idx = end_idx = None
            for index, character in enumerate(out_list):
                white_pix = np.count_nonzero(character.img)

                if start_idx is not None:
                    if white_pix == pixel_count:
                        end_idx = index
                    elif white_pix > pixel_count:
                        break

                if start_idx is None and white_pix >= pixel_count:
                    start_idx = index

            if start_idx is None:
                start_idx = index

            if end_idx is None:
                end_idx = index

            middle_idx = math.floor((start_idx + end_idx) / 2)

            self.white_pixel_indices[pixel_count] = middle_idx


        return out_list

    def calculate_fuzz_factor(self, size):
        fuzz_factor = 0

        if size < 2:
            fuzz_factor = 0
        elif size == 2:
            fuzz_factor = 3
        elif size == 3:
            fuzz_factor = 3
        elif size == 4:
            fuzz_factor = 6
        elif size >= 8:
            fuzz_factor = 8

        return fuzz_factor

    def convert_image(self, input_image):
        # Slice up the B&W input image into blocks and match each of them to an ASCII characters
        width, height = input_image.shape[1], input_image.shape[0]
        block_cols, block_rows = math.floor(width / self.char_width), math.floor(height / self.char_height)

        print(f" cols: {block_cols} rows: {block_rows}")

        self.used_chars = []  # Keep track of the unique characters used in this rendering, for analytics
        self.match_char_map = np.full((block_rows, block_cols), None, dtype=object)
        self.dist_eucl_map = np.full((block_rows, block_cols), None, dtype=object)
        self.pixel_diff_map = np.full((block_rows, block_cols), None, dtype=object)
        self.pixel_factor_map = np.full((block_rows, block_cols), None, dtype=object)
        self.hog_factor_map = np.full((block_rows, block_cols), None, dtype=object)
        self.candidate_chars = [[[] for i in range(block_cols)] for j in range(block_rows)]
        self.img_block_map = np.full((block_rows, block_cols), None, dtype=object)
        self.mask_map = np.full((block_rows, block_cols), None, dtype=object)


        output_image = input_image.copy()

        for row in range(0, block_rows):
            y = row * self.char_height

            for col in range(0, block_cols):
                x = col * self.char_width

                # Check whether block is inside the region to be converted
                if (self.region):
                    if (col not in range(*self.region[0])) or \
                            (row not in range(*self.region[1])):
                        continue

                img_block = input_image[y:y + self.char_height, x:x + self.char_width]
                self.img_block_map[row][col] = img_block

                matches = self.find_closest_matches(self.charset.chars, img_block)

                # Keep some character and image data about this conversion
                self.candidate_chars[row][col] = matches

                match, scores = self.get_best_match(matches, img_block)

                self.match_char_map[row][col] = match
                self.dist_eucl_map[row][col] = scores['dist_eucl']
                self.pixel_diff_map[row][col] = scores['pixel_diff']
                self.pixel_factor_map[row][col] = scores['pixel_factor']
                self.hog_factor_map[row][col] = scores['hog_factor']

                if (self.last_mask is not None):
                    self.mask_map[row][col] = self.last_mask

                self.used_chars.append(match)

                match_img = match.img
                # print(f"y: {y} x: {x}")
                output_image[y:y + self.char_height, x:x + self.char_width] = match_img

        # Return the list of Character objects used, along with the output image
        return output_image

    def find_closest_matches(self, characters, img):
        """
        Given a list of ASCII characters, and an image, find and return the charcter(s) which best match the pixels
        in the image
        """
        img_chr = Character(img)

        # Early check for empty or full images
        if(img_chr.is_full()):
            return [self.charset.full_char]
        elif(img_chr.is_empty()):
            return [self.charset.empty_char]

        white_pixel_count = np.count_nonzero(img)

        middle = self.get_middle_index(white_pixel_count)
        start = middle - math.ceil(self.num_close_matches / 2)
        if (start < 0):
            start = 0

        end = start + self.num_close_matches
        if (end > len(self.pixel_sorted_chars)):
            end = len(self.pixel_sorted_chars)

        matches = self.pixel_sorted_chars[start:end]

        return matches


    def get_best_match(self, charlist, img=None, max_acceptable=25):
        charlist = list(set(charlist)) # Dedupe

        if(len(charlist) == 1):
            return charlist[0], {
                "pixel_diff":0,
                "dist_eucl":0,
                "pixel_factor":0,
                "hog_factor":0
            }


        img_feat = self.get_image_features(img)
        total_pix = img.shape[0] * img.shape[1]
        half_pix = total_pix / 2

        white_pix = np.count_nonzero(img)
        black_pix = total_pix - white_pix

        img_weight = abs(white_pix - black_pix)


        # The pixel diff is weighted differently depending on the overall b/w balance of the source image
        if (img_weight < 16): # almost perfectly balanced white vs black pixels
            pixel_factor = 0.2
            hog_factor = 0.5
        elif ((img_weight >= 16) and (img_weight < 32)):
            pixel_factor = 1
            hog_factor = 2
        else: # very few white pixels or very few black pixels
            pixel_factor = 2
            hog_factor = 1

        lowest_dist = lowest_dist_eucl = lowest_pixel_diff = 100000000
        best_char = None

        # if(len(charlist) > 1):
        #     # Remove all empty and all full characters from consideration
        #     charlist = [character for character in charlist if (not character.is_full() and not character.is_empty())]

        for character in charlist:

            total_dist = 0
            pixel_dist = self.get_shifted_diff(img, character)
            dist_eucl = self.get_dist_with_char(img_feat, character, algo=distance.euclidean)

            total_dist += (dist_eucl) * hog_factor

            # Add the pixel diff to the HOG distance between image and character

            total_dist += pixel_dist * pixel_factor

            # print(f'eucl: {dist_eucl}/can:{dist_can}/pixel:{pixel_dist}')

            pixel_diff = self.get_pixel_diff(img, character.img)

            if total_dist < lowest_dist:
                best_char = character
                lowest_dist = total_dist
                lowest_pixel_diff = pixel_diff
                lowest_dist_eucl = dist_eucl

            elif total_dist == lowest_dist: # A tie!

                # First, try to pick the one with the lowest pixel diff
                if pixel_diff < lowest_pixel_diff:
                    best_char = character
                    lowest_dist = total_dist
                    lowest_pixel_diff = pixel_diff
                    lowest_dist_eucl = dist_eucl

                elif pixel_diff == lowest_pixel_diff: # Damnit, another tie!
                    # Using the character index seems pretty random, but at least it should be deterministic
                    # @TODO research better ways to find the best match in this edge case

                    if (character.index < best_char.index):
                        best_char = character
                        lowest_dist = total_dist
                        lowest_pixel_diff = pixel_diff

        return best_char, {
            "pixel_diff":lowest_pixel_diff,
            "dist_eucl":lowest_dist_eucl,
            "pixel_factor": pixel_factor,
            "hog_factor": hog_factor
        }


    def get_image_features(self, image, visualize=False):
        """ Get the HOG (histogram of oriented gradients) for the image passed, which should be in standard
         OpenCV format. Will return either the feature descriptor, or a tuple of the features and the image
        with the gradients depending on the parameter 'visualize' """

        image = cv.resize(image, (64, 64), interpolation=cv.INTER_NEAREST_EXACT)

        # Convert image from OpenCV format to SciKit image usable format
        image = util.img_as_ubyte(image)

        features = feature.hog(
            image,
            orientations=5,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=visualize
        )

        return features

    def get_dist_with_char(self, image_feat, character, algo: Callable=None):
        # Algo should be one of:
        #
        # distance.euclidean
        # distance.braycurtis

        char_feat = self.char_feat_map[character.index]

        dist = algo(image_feat, char_feat)

        if math.isnan(dist):
            dist = 0

        return dist

    def calculate_character_features(self):
        """Precalculate image feature descriptors via HOG for each of the character images"""
        for character in self.charset:
            index = character.index
            feat = self.get_image_features(character.img)

            self.char_feat_map[index] = feat


    def get_shifted_diff(self, img, character):
        """Returns the lowest diff across a series of variant comparisons made by shifting
        one of the images in a few directions"""

        diffs = []

        # Fabricate coordinate pairs
        for char in self.shifted_chars[character.index]:
            # If the character is empty/full, it's not going to be a good match
            # if tools.is_empty(char) or tools.is_full(char):
            #     continue

            diff = self.get_pixel_diff(char, img)

            # weight the diff based on how far we are from the original center
            # dist = abs(params[0]) + abs(params[1])
            dist = 1
            dist_weight = 0.1
            diff = diff * (dist * dist_weight)
            diffs.append(diff)

        return min(diffs)


    def get_start_end_index(self, count):
        """Returns a tuple of the start and end index of the character block which has the specified white pixel count"""
        start, end = None, None

        for index, character in enumerate(self.pixel_sorted_chars):
            char_pixels = np.count_nonzero(character.img)

            if (char_pixels >= count) and (start is None):
                start = index
            if (char_pixels > count) and (end is None):
                end = index

        if end is None:
            end = len(self.pixel_sorted_chars) - 1

        return start, end


    def get_middle_index(self, count):
        if count > len(self.white_pixel_indices) - 1:
            count = len(self.white_pixel_indices) - 1

        middle_index = self.white_pixel_indices[count]
        return  middle_index


    def get_weighted_diff(self, img, character, weights):
        """ Calculate the pixel difference between character and image, but taking into account the weights
        passed along as a separate image. White pixels in the weight image indicate high relevance diffs and black
        pixels indicate low relevance diffs"""

        diff = np.abs(img - character.img)
        diff = diff * weights

        return np.sum(diff)

    def generate_shifted_chars(self):
        """ Pregenerate pixel shifted versions of each character, which will be used when matching against
        an image for close matches"""

        shifted_chars = {}

        for char in self.charset.chars:
            char_list = []

            for params in itertools.product(self.shift_dims, self.shift_dims):
                shifted_char = self.shift_img(char.img, params[0], params[1])
                char_list.append(shifted_char)

            shifted_chars[char.index] = char_list

        self.shifted_chars = shifted_chars






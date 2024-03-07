import cv2 as cv
import numpy as np

from ascii import AsciiMatcher

class EdgeMatcher(AsciiMatcher):

    def __init__(self, characters=None, charset=None):
        super(EdgeMatcher, self).__init__(characters)
        self.charset = charset

    def find_match(self, img):
        pixel_threshold = 8

        # Filter out images with very few white or black pixels, which are not likely to do well with heartbeat matching

        # Almost black images
        if  (self.white_pixel_count(img) <= pixel_threshold):
            matches = self.find_strict_match(img)

            if len(matches) == 0:
                return [self.charset.empty_char]

            return matches

        # Almost white images
        if (self.black_pixel_count(img) <= pixel_threshold):
            matches = self.find_strict_match(img)

            if len(matches) == 0:
                return [self.charset.full_char]

            return matches


        num = len(self.characters)
        # print(f"Looking for match among {num} characters")

        prev_matches = new_matches = self.characters

        # Start at a high fuzz factor and decrease if we find more than one match
        for fuzz_factor in range(8,0):
            new_matches = self.match_by_heartbeat(img, fuzz_factor)

            if len(new_matches == 1):
                return new_matches
            elif len(new_matches) == 0:
                # We went too far, return the previous matches
                new_matches = prev_matches
                break

            prev_matches = new_matches


        return new_matches


    def find_strict_match(self, img):
        best_matches = []
        diffs = []

        fuzz_factor = 2

        for character in self.characters:
            this_diff = self.get_pixel_diff(character.img, img)
            diffs.append(this_diff)

        lowest_diff = 0

        for diff, match in zip(diffs, self.characters):
            delta = diff - lowest_diff

            if delta <= fuzz_factor:
                best_matches.append(match)

        return best_matches

    def match_by_heartbeat(self, img, fuzz_factor=8):
        """Dilates, then erodes the image as well as each possible match character in order to find better matches"""
        inverse = True if np.count_nonzero(img) > 32 else False
        img = self.heartbeat_transform(img, inverse)

        best_matches = []
        diffs = []

        for character in self.characters:
            transformed = self.heartbeat_transform(character.img)
            this_diff = self.get_pixel_diff(transformed, img)
            diffs.append(this_diff)

        lowest_diff = min(diffs)

        for diff, match in zip(diffs, self.characters):
            delta = diff - lowest_diff

            if delta <= fuzz_factor:
                best_matches.append(match)

        return best_matches


    def heartbeat_transform(self, img, inverse=False):
        if inverse:
            img = self.topo_erode(img)
            img = self.topo_dilate(img)
        else:
            img = self.topo_dilate(img)
            img = self.topo_erode(img)

        return img

    def topo_dilate(self, img):
        img = img.copy()

        matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        kernel = np.array(matrix, np.uint8)

        img = cv.dilate(img, kernel)
        return img

    def topo_erode(self, img):
        img = img.copy()

        matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        kernel = np.array(matrix, np.uint8)

        img = cv.erode(img, kernel)
        return img

    def white_pixel_count(self, img):
        return np.count_nonzero(img)

    def black_pixel_count(self, img):
        total_pixels = img.shape[0] * img.shape[1]
        return total_pixels - np.count_nonzero(img)


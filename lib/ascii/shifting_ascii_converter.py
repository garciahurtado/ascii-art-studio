import itertools
from charset import Charset, Character
from ascii import AsciiConverter
import numpy as np
import cv2 as cv
from cvtools import size_tools as tools

class ShiftingAsciiConverter(AsciiConverter):
    """ASCII converter which allows for the selection of specific characters from the range of
    original candidates"""

    def __init__(self, charset:Charset):
        super(ShiftingAsciiConverter, self).__init__(charset)

    def calculate_fuzz_factor(self, size):
        fuzz_factor = 0

        if size < 2:
            fuzz_factor = 0
        elif size == 2:
            fuzz_factor = 3
        elif size == 3:
            fuzz_factor = 3
        elif size == 4:
            fuzz_factor = 7
        elif size >= 8:
            fuzz_factor = 10

        return fuzz_factor


    def find_closest_matches(self, characters, img, size):
        """
        Given a list of ASCII characters, and an image, find and return the character(s) which best match the pixels
        in the image
        """
        img_chr = Character(img)

        # Early check for empty or full images
        if(img_chr.is_full()):
            return [self.charset.full_char]
        elif(img_chr.is_empty()):
            print("EMPTY!")
            return [self.charset.empty_char]

        prev_diff = size * size + 1
        close_matches = []

        fuzz_factor = self.calculate_fuzz_factor(size)

        if size <= 8:
            img_tuple = tuple(img.flatten().tolist())
        else:
            img_tuple = None

        for possible_match in characters:
            # Count the number of pixels which are different between the image chunk and the current character

            if (img_tuple):
                this_diff = Character.diff_cached(possible_match.img_tuple, img_tuple, size)
            else:
                this_diff = Character.diff(possible_match.img, img)

            possible_match.match_diff = this_diff
            next_lower = int(size / 2)

            # Accumulate diff from parent low res character, but only if the diff is more than the fuzz factor
            if ((this_diff > fuzz_factor) and (next_lower in possible_match.low_res)):
                if (size == 8):
                    # We add it a second time, since errors at the highest res are the most significant
                    possible_match.match_diff += this_diff / 2
                else:
                    possible_match.match_diff += possible_match.low_res[next_lower].match_diff * (size / 2)

            diff_delta = this_diff - prev_diff

            if diff_delta > fuzz_factor:
                # The new match is worse, forget it
                continue

            if diff_delta < - fuzz_factor:
                # We found a better match than the last one, so forget about the last one and keep track of it
                # close_matches = []
                close_matches.append(possible_match)
                prev_diff = this_diff

            elif abs(diff_delta) <= fuzz_factor:
                # Both this match and the previous one are about as good
                close_matches.append(possible_match)

        return close_matches

    def get_best_match(self, charlist, img=None, max_acceptable = 25):
        """ Given a list of already matched characters, return the one with the lowest diff score

        charlist - a list of Characters to search
        img - the image to find the best character match for
        max_acceptable - The lowest bound of the acceptable diff between img and character. Higher than this value
        will trigger looking through 'shifted' versions of the character for a better match.
        """
        final_match = charlist[0]


        if (len(charlist) == 1):
            return charlist[0]
        else:
            lowest_diff = self.char_width * self.char_width

            for best_match in charlist:
                if best_match.match_diff <= lowest_diff:
                    final_match = best_match
                    lowest_diff = best_match.match_diff

        if lowest_diff > max_acceptable:
            """Match against pixel shifted versions of the original character list"""
            diffs = []
            shift_dims = (2, 1, 0, -1, -2)

            for params in itertools.product(shift_dims, shift_dims):
                shifted = self.shift_chars(charlist, params)

                for shifted_img in shifted:
                    # If the character is empty/full after shifting, it's not going to be a good match
                    if tools.is_empty(shifted_img) or tools.is_full(shifted_img):
                        continue

                    diff = self.get_pixel_diff(shifted_img, img)
                    diffs.append(diff)

            best_diff = 1000
            best_index = 0
            for diff, index in zip(diffs, itertools.cycle(range(0,len(charlist)))):
                if diff < best_diff:
                    best_index = index
                    best_diff = diff

            final_match = charlist[best_index]

        return final_match


    def shift_chars(self, charlist, delta_shift):
        output = []
        delta_x, delta_y = delta_shift

        for character in charlist:
            shifted = self.shift_img(character.img, delta_x, delta_y)
            output.append(shifted)

        return output


    def shift_img(self, img, delta_x, delta_y):
        translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
        output_img = cv.warpAffine(img, translation_matrix, [img.shape[1], img.shape[0]], borderMode=cv.BORDER_REPLICATE)
        return output_img

    def find_strict_match(self, matches, img, fuzz_factor=2):
        best_matches = []
        diffs = []

        for character in matches:
            this_diff = self.get_pixel_diff(character.img, img)
            diffs.append(this_diff)

        lowest_diff = 0

        for diff, match in zip(diffs, matches):
            delta = diff - lowest_diff

            if delta <= fuzz_factor:
                best_matches.append(match)

        return best_matches


    def select_best_by_pixel_sum(self, matches, img):

        best_diff = 1000
        best_matches = []

        for character in matches:
            img_sum = np.sum(img) / 64
            char_sum = np.sum(character.img) / 64
            diff = abs(img_sum - char_sum)

            if diff == best_diff:
                best_matches.append(character)
            elif diff < best_diff:
                best_matches = [character]
                best_diff = diff

        return best_matches

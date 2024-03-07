import os

import cv2 as cv

from ascii import AsciiMatcher

class MaskMatcher(AsciiMatcher):
    MASK_DIR = os.path.dirname(__file__) + '/res/masks/'

    def __init__(self, characters=None):
        super(MaskMatcher, self).__init__(characters)

        self.filters = []
        self.last_mask = None

    def add_mask(self, mask_name, fuzz=0, blanks=False, filtering=False, maxdiff=8):
        mask_filter = {}
        mask_filter['name'] = mask_name
        mask_filter['mask'] = self.load_mask(mask_name)
        mask_filter['fuzz_factor'] = fuzz
        mask_filter['allow_blank'] = blanks
        mask_filter['filtering'] = filtering
        mask_filter['maxdiff'] = maxdiff

        self.filters.append(mask_filter)


    def load_mask(self, name):
        filename = self.MASK_DIR + name + '.png'

        if not os.path.exists(filename):
            raise FileExistsError(f'Mask file {filename} does not exist')

        mask = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        return mask


    def find_match(self, img):
        """Go through a list of masks, one at a time, and pass them to select_best_by_mask. If any of them
        return a single result, return that character. Otherwise, continue through the list."""
        self.img = img
        i = 0

        self.last_mask = None

        for filter in self.filters:
            fuzz_factor = filter['fuzz_factor']
            name = filter['name']
            mask = filter['mask']
            allow_blank = filter['allow_blank']
            filtering = filter['filtering']
            maxdiff = filter['maxdiff']

            results = self.filter_by_mask(mask, fuzz_factor=fuzz_factor, allow_blank=allow_blank, filtering=filtering, maxdiff=maxdiff)
            i += 1

            if len(results) == 1:
                self.last_mask = mask
                # print(f'Applied "{name}" mask to reduce to single match')
                return results

        return self.characters


    def filter_by_mask(self, mask, fuzz_factor=0, allow_blank=False, filtering=False, maxdiff=8):
        """Using a mask to isolate the most important pixels in a block, count the differences in each
        candidate character in order to select the best one.

        Please note that if more than one character have the same pixel matching score, multiple characters
        could be returned
        """

        best_matches = []
        diffs = []

        for character in self.characters:
            this_diff = self.get_diff_with_mask(character, mask)
            diffs.append(this_diff)

        lowest_diff = min(diffs)

        for diff, match in zip(diffs, self.characters):
            if not allow_blank and (match.is_full() or match.is_empty()):
                continue

            if diff > maxdiff:
                continue

            delta = diff - lowest_diff

            if delta <= fuzz_factor:
                best_matches.append(match)

        if(len(best_matches) > 1 and filtering):
            self.characters = best_matches

        return best_matches


    def get_diff_with_mask(self, character, mask):
        """
        Using a mask, determine which pixels are the same between the character and the image
        """
        masked_img = cv.bitwise_and(self.img, mask)
        masked_char = cv.bitwise_and(character.img, mask)
        diff = self.get_pixel_diff(masked_img, masked_char)

        return diff







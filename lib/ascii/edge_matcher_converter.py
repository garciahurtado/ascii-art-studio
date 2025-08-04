import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from charset import Charset, Character
from ascii.shifting_ascii_converter import ShiftingAsciiConverter
from ascii.edge_matcher import EdgeMatcher
import cv2 as cv


class EdgeMatcherConverter(ShiftingAsciiConverter):
    """ASCII converter that uses edge detection and morphological operations for matching.

    This converter uses the EdgeMatcher to find the best character matches based on
    edge similarity and morphological transformations.
    """

    def __init__(self, charset: Charset):
        """Initialize the EdgeMatcherConverter with a character set.

        Args:
            charset: The character set to use for conversion
        """
        super(EdgeMatcherConverter, self).__init__(charset)

        # Initialize the edge matcher
        self.edge_matcher = EdgeMatcher(charset=charset)

        # Initialize match tracking
        self.match_char_map = None
        self.candidate_chars = None
        self.img_block_map = None
        self.pixel_diff_map = None
        self.used_chars = []

    def convert_image(self, input_image: np.ndarray) -> np.ndarray:
        """Convert an input image to ASCII art using edge-based matching.

        Args:
            input_image: Input grayscale image (0-255)

        Returns:
            Grayscale image with ASCII characters
        """
        height, width = input_image.shape
        block_cols = width // self.char_width
        block_rows = height // self.char_height

        # Initialize output image and tracking structures
        output_image = np.zeros_like(input_image)
        self.used_chars = []
        self.match_char_map = np.full((block_rows, block_cols), None, dtype=object)
        self.candidate_chars = np.full((block_rows, block_cols), None, dtype=object)
        self.img_block_map = np.full((block_rows, block_cols), None, dtype=object)
        self.pixel_diff_map = np.full((block_rows, block_cols), None, dtype=float)

        # Process each block
        for row in range(block_rows):
            for col in range(block_cols):
                y = row * self.char_height
                x = col * self.char_width

                # Extract block
                img_block = input_image[y:y + self.char_height, x:x + self.char_width]
                self.img_block_map[row, col] = img_block

                # Find matches using EdgeMatcher
                matches = self.edge_matcher.find_match(img_block)
                self.candidate_chars[row, col] = matches

                # Get best match
                match = self._get_best_match(matches, img_block)
                self.match_char_map[row, col] = match

                # Store diff for analysis
                diff = self.get_pixel_diff(match.img, img_block)
                self.pixel_diff_map[row, col] = diff

                # Add to used chars if not already present
                if match not in self.used_chars:
                    self.used_chars.append(match)

                # Place character in output
                output_image[y:y + self.char_height, x:x + self.char_width] = match.img

        return output_image

    def _get_best_match(self, matches: List[Character], img_block: np.ndarray) -> Character:
        """Get the best match from a list of candidates.

        Args:
            matches: List of candidate characters
            img_block: Input image block to match against

        Returns:
            Best matching character
        """
        if not matches:
            return self.charset.empty_char

        if len(matches) == 1:
            return matches[0]

        # If multiple matches, use pixel difference to break ties
        best_match = None
        best_diff = float('inf')

        for match in matches:
            diff = self.get_pixel_diff(match.img, img_block)
            if diff < best_diff:
                best_diff = diff
                best_match = match

        return best_match

    def get_used_chars(self) -> List[Character]:
        """Get the list of characters used in the last conversion.

        Returns:
            List of Character objects used in the last conversion
        """
        return self.used_chars

    def get_match_map(self) -> np.ndarray:
        """Get the character match map from the last conversion.

        Returns:
            2D numpy array of Character objects
        """
        return self.match_char_map

    def get_candidate_map(self) -> np.ndarray:
        """Get the candidate characters considered for each position.

        Returns:
            2D numpy array of lists of Character objects
        """
        return self.candidate_chars

    def get_pixel_diff_map(self) -> np.ndarray:
        """Get the pixel difference map from the last conversion.

        Returns:
            2D numpy array of pixel difference values
        """
        return self.pixel_diff_map
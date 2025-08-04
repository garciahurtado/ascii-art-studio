"""An image processing pipeline which executes functions sequentially on an image input,
passing the output to the next function in the pipeline until all have been executed.

The order in which the '_run' methods are defined in the class is the order in which they will be run"""

import numpy as np
import cv2 as cv
import cvtools.size_tools as tools
import cvtools.contrast_filters as filters
from color.palette_extractor import PaletteExtractor
from cvtools import color_filters
from logging_config import logger

class ProcessingPipeline():
    # Tunable pipeline parameters
    brightness = 0
    contrast = 0

    def __init__(self, brightness=0, contrast=1):
        """Initialize the processing pipeline with optional brightness and contrast settings.
        
        Args:
            brightness: Brightness adjustment (0-100, default 0, no change)
            contrast: Contrast adjustment (1.0-3.0, default 1.0 - no change)
        """
        # Brightness is already in 0-100 range for OpenCV
        # self.brightness = -400 + (float(brightness) * 8.0)  # -400 -> +400
        self.brightness = brightness

        # Map contrast from 0-100 to 1.0-3.0 (1.0 is no change, 3.0 is triple contrast)
        # self.contrast = 1.0 + (float(contrast) / 100.0 * 4.0)  # 0-100 -> 1.0-5.0
        self.contrast = contrast

        logger.info(f"Pipeline initialized with Brightness: {self.brightness}, Contrast: {self.contrast}")
        
        # Initialize other instance variables
        self.img_width = None
        self.img_height = None
        self.char_height = 8
        self.char_width = 8
        self.converter = None
        self.grayscale = None
        self.contrast_mask = None
        self.color_mask = None
        self.contrast_img = None
        self.ascii = None
        self.ascii_inv = None
        self.fg_colors = None
        self.bg_colors = None
        self.color_ascii = None
        self.palette = None
        self.extractor = None
        self.invert = None
        self.color = None
        self.original = None

    def run(self, input_img, invert=False):
        self.original = input_img
        self.color = self.original
        self.extractor = PaletteExtractor()
        self.invert = invert

        steps = []

        class_members = self.__class__.__dict__.keys()

        # Collect the names of all class methods that start with '_run'
        for method_name in class_members:
            if method_name.startswith('_run'):
                steps.append(method_name)

        output_img = input_img

        # Run them one at a time, in the order they appear in the source code, passing the output of one method as the
        # input to the next
        for step in steps:
            run_step = getattr(self, step)
            output_img = run_step(input_img)
            input_img = output_img

        return output_img

    """Disabled for video, since we resize at frame capture time"""
    def _run_resize_with_padding(self, input_img):
        actual_height = input_img.shape[0]
        actual_width = input_img.shape[1]

        # Don't resize unnecessarily
        if(actual_height != self.img_height) or (actual_width != self.img_width):
            output_img = tools.resize_with_padding(input_img, (self.img_width, self.img_height))
            self.original = output_img
        else:
            self.original = output_img = input_img

        self.color = self.original = output_img
        return output_img


    def _run_create_grayscale(self, input_img):
        grayscale = input_img.copy()
        grayscale = cv.cvtColor(grayscale, cv.COLOR_BGR2GRAY)
        grayscale = color_filters.brightness_contrast(grayscale, self.brightness, self.contrast)
        self.grayscale = grayscale
        return input_img


    def _run_create_high_contrast(self, input_img):
        # This used to be (char_height * 2, char_width * 2), but it was too much detail which was degrading the final
        # output. After visually comparing 0.5x, 1x and 2x, 1x was the best. This does not affect output size or the
        # number of ASCII characters in the final image, only the size of the blocks that are used for creating contrast

        block_size = (self.char_height, self.char_width)
        self.contrast_img = filters.block_contrast(self.grayscale, block_size, invert=self.invert)
        return input_img

    def _run_denoise(self, input_img):
        contrast = self.contrast_img

        # Use the "hit or miss" algorithm to get rid of noise:
        kernel = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype="int")
        strays = cv.morphologyEx(contrast, cv.MORPH_HITMISS, kernel)
        self.contrast_img = cv.bitwise_xor(strays, contrast)
        # self.contrast_img = strays
        return input_img


    def _run_convert_to_ascii(self, img):
        self.ascii = self.converter.convert_image(self.contrast_img)
        self.ascii_inv = cv.bitwise_not(self.ascii)
        return self.ascii


    def _run_extract_colors_from_mask(self, mask_img):
        char_size = [self.char_width, self.char_height]
        self.fg_colors = color_filters.extract_colors_from_mask(self.color, mask_img, self.contrast_img, char_size)
        self.bg_colors = color_filters.extract_colors_from_mask(self.color, mask_img, self.contrast_img, char_size, invert=True)

        return mask_img


    def _run_color_ascii(self, input_img):
        fg_colors_big = cv.resize(self.fg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)
        bg_colors_big = cv.resize(self.bg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)

        # Use ASCII as a mask for foreground colors
        self.ascii = cv.cvtColor(self.ascii, cv.COLOR_GRAY2BGR)
        fg_color = cv.bitwise_and(self.ascii, fg_colors_big)

        # Use ASCII as a mask for background colors
        self.ascii_inv = cv.cvtColor(self.ascii_inv, cv.COLOR_GRAY2BGR)
        bg_color = cv.bitwise_and(self.ascii_inv, bg_colors_big)

        self.color_ascii = cv.bitwise_or(bg_color, fg_color)

        # Convert back from CV BGR
        true_bg_colors = cv.cvtColor(self.bg_colors, cv.COLOR_BGR2RGB)
        true_fg_colors = cv.cvtColor(self.fg_colors, cv.COLOR_BGR2RGB)

        # save the colors in the blocks of the converter
        for row_idx in range(len(self.converter.match_char_map)):
            row = self.converter.match_char_map[row_idx]
            for col_idx in range(len(row)):
                block = row[col_idx]
                block.bg_color = true_bg_colors[row_idx][col_idx]
                block.fg_color = true_fg_colors[row_idx][col_idx]

        # self.color_ascii = color_filters.palettize(self.color_ascii, self.palette)

        return bg_color


    def _run_block_colors(self, _):
        # Resize to img / character-size since we only need one color per 8x8 block
        dims = (int(self.img_width / self.char_width), int(self.img_height / self.char_height))
        flat_colors = cv.resize(self.color, dims, interpolation=cv.INTER_LANCZOS4)

        # Resize back up to original size
        flat_colors = cv.resize(flat_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST_EXACT)
        flat_colors_masked = cv.bitwise_and(flat_colors, flat_colors, mask=self.color_mask)

        self.flat_colors = flat_colors

        return _


    def _run_final_blend(self, _):
        """Final mix between colored ASCII blocks and flat color blocks"""
        # blended = cv.bitwise_or(self.color_ascii, self.flat_colors)
        self.color_ascii = color_filters.quantize_img(self.color_ascii)
        return self.color_ascii

    # DISABLED
    def __run_final_resize(self, input_img):
        """Final resize, enlarging x2 without antialias"""
        final = cv.resize(input_img, (input_img.shape[1] * 2, input_img.shape[0] * 2), interpolation=cv.INTER_NEAREST)

        return final

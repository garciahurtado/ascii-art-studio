"""An image processing pipeline which executes functions sequentially on an image input,
passing the output to the next function in the pipeline until all have been executed.

The order in which the '_run' methods are defined in the class is the order in which they will be run"""

import numpy as np
import cv2 as cv
import cvtools.size_tools as tools
import cvtools.contrast_filters as filters
from ascii import AsciiConverter
from color.palette_extractor import PaletteExtractor
from cvtools import color_filters


class ProcessingPipeline():
    img_width = None
    img_height = None
    char_height = 8
    char_width = 8
    converter:AsciiConverter = None

    # Images at each stage of the pipeline
    grayscale = None
    contrast_mask = None
    color_mask = None
    contrast_img = None
    ascii = None
    ascii_inv = None
    fg_colors = None
    bg_colors = None

    palette = None
    extractor = None

    def run(self, input_img, invert=False):
        self.original = input_img
        self.extractor = PaletteExtractor()
        self.invert = invert

        steps = []

        class_members = self.__class__.__dict__.keys()

        # Collect the names of all class methods that start with '_run'
        for method_name in class_members:
            if method_name.startswith('_run'):
                steps.append(method_name)

        output_img = input_img

        # Run them one at a time, passing the output of one method as the input to the next
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
            output_img = input_img

        self.color = self.original
        return output_img


    def _run_brightness_saturation(self, input_img):
        #output_img = color_filters.brightness_saturation(input_img, 1.1, 1.4)
        self.color = self.original
        self.original = input_img

        return input_img


    def _run_create_grayscale(self, input_img):
        grayscale = input_img.copy()
        # grayscale = colors.brightness_contrast(grayscale, 20, 20)
        self.grayscale = cv.cvtColor(grayscale, cv.COLOR_BGR2GRAY)

        return input_img


    def _run_create_high_contrast(self, input_img):
        self.contrast_img = filters.block_contrast(self.grayscale, (self.char_height, self.char_width), invert=self.invert)

        return input_img


    def _run_denoise(self, input_img):
        contrast = self.contrast_img

        # Use the "hit or miss" algorithm to get rid of noise:
        kernel = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype="int")
        strays = cv.morphologyEx(contrast, cv.MORPH_HITMISS, kernel)
        self.contrast_img  = cv.bitwise_xor(strays, contrast)
        # self.contrast_img = strays
        return input_img


    def _run_convert_to_ascii(self, img):
        self.ascii = self.converter.convert_image(self.contrast_img)
        self.ascii_inv = cv.bitwise_not(self.ascii)

        return self.contrast_img


    def _run_extract_colors_from_mask(self, input_img):
        char_size = [self.char_width, self.char_height]
        fg_colors = color_filters.extract_colors_from_mask(self.color, input_img, char_size)
        bg_colors = color_filters.extract_colors_from_mask(self.color, input_img, char_size, invert=True)

        self.fg_colors = fg_colors
        self.bg_colors = bg_colors

        return input_img


    def _run_color_ascii(self, input_img):
        fg_colors_big = cv.resize(self.fg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)
        bg_colors_big = cv.resize(self.bg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)

        # Use ASCII as a mask for foreground colors
        ascii = cv.cvtColor(self.ascii, cv.COLOR_GRAY2BGR)
        fg_ascii = cv.bitwise_and(fg_colors_big, ascii)

        # Use ASCII as a mask for background colors
        ascii_inv = cv.cvtColor(self.ascii_inv, cv.COLOR_GRAY2BGR)
        bg_ascii = cv.bitwise_and(bg_colors_big, ascii_inv)

        self.color_ascii = cv.bitwise_or(bg_ascii, fg_ascii)

        return self.color_ascii


    def _run_block_colors(self, input_img):
        # Resize to img / character-size since we only need one color per 8x8 block
        dims = (int(self.img_width / self.char_width), int(self.img_height / self.char_height))
        flat_colors = cv.resize(self.original, dims, interpolation=cv.INTER_LINEAR)

        # Resize back up to original size
        flat_colors = cv.resize(flat_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)
        flat_colors_masked = cv.bitwise_and(flat_colors, flat_colors, mask=self.color_mask)

        self.flat_colors = flat_colors_masked

        return input_img


    def _run_final_blend(self, input_img):
        """Final mix between colored ASCII blocks and flat color blocks"""
        blended = cv.bitwise_or(input_img, self.flat_colors)
        # blended = input_img

        return blended

    def _run_final_resize(self, input_img):
        """Final resize, enlarging x2 without antialias"""
        final = cv.resize(input_img, (input_img.shape[1] * 2, input_img.shape[0] * 2), interpolation=cv.INTER_NEAREST)

        return final

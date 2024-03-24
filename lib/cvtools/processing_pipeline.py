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
    color_ascii = None

    palette = None
    extractor = None
    invert = None
    color = None

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
            self.original = output_img = input_img

        self.color = self.original = output_img
        return output_img


    def _run_create_grayscale(self, input_img):
        grayscale = input_img.copy()
        # grayscale = colors.brightness_contrast(grayscale, 20, 20)
        self.grayscale = cv.cvtColor(grayscale, cv.COLOR_BGR2GRAY)

        return input_img


    def _run_create_high_contrast(self, input_img):
        self.contrast_img = filters.block_contrast(self.grayscale, (self.char_height*2, self.char_width*2), invert=self.invert)

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
        inv_contrast = cv.bitwise_not(self.contrast_img)
        self.fg_colors = color_filters.extract_colors_from_mask(self.color, mask_img, self.contrast_img, char_size)
        self.bg_colors = color_filters.extract_colors_from_mask(self.color, mask_img, self.contrast_img, char_size, invert=True)

        return mask_img


    def _run_color_ascii(self, input_img):
        fg_colors_big = cv.resize(self.fg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)
        bg_colors_big = cv.resize(self.bg_colors, (self.img_width, self.img_height), interpolation=cv.INTER_NEAREST)

        # Use ASCII as a mask for foreground colors
        self.ascii = cv.cvtColor(self.ascii, cv.COLOR_GRAY2BGR)
        fg_ascii = cv.bitwise_and(self.ascii, fg_colors_big)

        # Use ASCII as a mask for background colors
        self.ascii_inv = cv.cvtColor(self.ascii_inv, cv.COLOR_GRAY2BGR)
        bg_ascii = cv.bitwise_and(self.ascii_inv, bg_colors_big)

        self.color_ascii = cv.bitwise_or(bg_ascii, fg_ascii)
        # self.color_ascii = color_filters.palettize(self.color_ascii, self.palette)

        return bg_ascii


    def __run_block_colors(self, _):
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
        #blended = cv.bitwise_or(self.color_ascii, self.flat_colors)
        #self.color_ascii = color_filters.quantize_img(self.color_ascii)
        return self.color_ascii

    def _run_final_resize(self, input_img):
        """Final resize, enlarging x2 without antialias"""
        final = cv.resize(input_img, (input_img.shape[1] * 2, input_img.shape[0] * 2), interpolation=cv.INTER_NEAREST)

        return final

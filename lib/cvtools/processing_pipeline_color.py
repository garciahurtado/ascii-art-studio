"""An image processing pipeline that extends the full pipeline to only run:
- Splitting B&W ASCII image into foreground/background contrast images (inverse of one another)
- For each contrast image:
    - Use it as a mask over full color image
    - Extract colors and resize to standard color block
- Use both color images (FG and BG) to extract limited perceptual palette (using Kmeans)
- Apply palette to the color frames and return them both combined into one

"""
import cv2 as cv

from color.palette_extractor import PaletteExtractor
from . import color_filters
from .processing_pipeline import ProcessingPipeline

class ProcessingPipelineColor(ProcessingPipeline):
    def __init__(self, brightness=0, contrast=0):
        """Initialize the color processing pipeline with optional brightness and contrast settings.
        
        Args:
            brightness: Brightness adjustment (0-100, default 0)
            contrast: Contrast adjustment (0-100, mapped to 1.0-3.0, default 1.0)
        """
        super().__init__(brightness, contrast)
        self.extractor = PaletteExtractor()

    def run(self, input_image):
        self.original = input_image
        self.color = self.original

        image = self._run_resize_with_padding(input_image)
        image = self._run_create_grayscale(image)
        image = self._run_create_high_contrast(image)
        image = self._run_denoise(image)
        image = self._run_convert_to_ascii(image)
        image = self._run_extract_colors_from_mask(image)
        image = self._run_color_ascii(image)
        image = self._run_block_colors(image)
        image = self._run_final_blend(image)
        image = self.resize(image)

        return image

    def extract_palette(self, image, num_colors=64, previous_palette=None):
        color_idx, self.palette = self.extractor.extract_palette(
            image,
            num_colors=num_colors,
            previous_palette=previous_palette)

        return color_idx, self.palette


    def palettize(self, fg_colors, bg_colors, color_idx):
        # Divide color indices: half for foreground colors and half for background colors
        color_idx_fg = color_idx[0:int(len(color_idx) / 2)]
        color_idx_bg = color_idx[int(len(color_idx) / 2):len(color_idx)]

        # Apply the palette to both the foreground and background color images
        self.fg_color_idx, self.fg_colors = color_filters.palettize(fg_colors, self.palette, color_idx=color_idx_fg)
        self.bg_color_idx, self.bg_colors = color_filters.palettize(bg_colors, self.palette, color_idx=color_idx_bg)

        return [self.fg_colors, self.bg_colors]

    def compose(self, ascii):
        '''Once the colors have been palettized, and the ASCII B&W prerendered, run the final composition'''
        self.ascii = ascii
        self.ascii_inv = cv.bitwise_not(ascii)

        image = self._run_color_ascii(ascii)
        return image


    def resize(self, image):
        image = self._run_final_resize(image)
        return image
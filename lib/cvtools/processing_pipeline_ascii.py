"""An image processing pipeline that extends the full pipeline to only run:
- Convert to black and white
- Convert to ASCII with Pytorch model
- Extract colors from ASCII mask
"""

from .processing_pipeline import ProcessingPipeline

class ProcessingPipelineAscii(ProcessingPipeline):

    def run(self, image):
        self.original = self.color = image

        image = self._run_resize_with_padding(image)
        image = self._run_create_grayscale(image)
        image = self._run_create_high_contrast(image)
        image = self._run_denoise(image)
        self._run_convert_to_ascii(image)
        self._run_extract_colors_from_mask(self.ascii)
        #self._run_final_resize

        return self.ascii
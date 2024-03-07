"""An image processing pipeline that extends the full pipeline to only run:
- Convert to black and white
- Convert to ASCII with ML
"""

from .processing_pipeline import ProcessingPipeline

class ProcessingPipelineAscii(ProcessingPipeline):
    def run(self, image):
        self.original = image

        image = self._run_brightness_saturation(image)
        image = self._run_create_grayscale(image)
        image = self._run_create_high_contrast(image)
        image = self._run_denoise(image)
        image = self._run_convert_to_ascii(image)

        return [self.ascii, self.contrast_img, self.color]
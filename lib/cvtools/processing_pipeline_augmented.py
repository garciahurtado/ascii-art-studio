"""An image processing pipeline which executes functions sequentially on an image input,
passing the output to the next function in the pipeline until all have been executed.

The order in which the '_run' methods are defined in the class is the order in which they will be run"""

import numpy as np
import cv2 as cv
import cvtools.size_tools as tools
import cvtools.contrast_filters as filters
from color.palette_extractor import PaletteExtractor
from cvtools import color_filters
from cvtools.processing_pipeline import ProcessingPipeline
from logging_config import logger


class ProcessingPipelineAugmented(ProcessingPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        print(steps)
        exit(1)

        output_img = input_img

        # Run them one at a time, in the order they appear in the source code, passing the output of one method as the
        # input to the next
        for step in steps:
            run_step = getattr(self, step)
            output_img = run_step(input_img)
            input_img = output_img

        return output_img

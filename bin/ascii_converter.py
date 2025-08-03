import sys

import cv2 as cv

from ascii import NeuralAsciiConverterPytorch, FeatureAsciiConverter, ShiftingAsciiConverter, AsciiConverter
from charset import Charset
from cvtools.processing_pipeline import ProcessingPipeline
from cvtools.processing_pipeline_ascii import ProcessingPipelineAscii
from logging_config import logger


# Initialize the converter with C64 charset
CHAR_WIDTH, CHAR_HEIGHT = 8, 8
CHARSET_NAME = 'c64.png'
MODEL_FILENAME = 'ascii_c64-Mar17_21-33-46'
MODEL_CHARSET = 'ascii_c64'
PALETTE_NAME = 'atari.png'

def convert_image(input_img, charset: Charset, img_width, img_height, converter_class=None):
    """"
    This converter takes an image path and converts it to a PNG file using ASCII character replacement.
    """
    image = cv.imread(input_img)
    converter = init_converter(charset, converter_class)

    # The brightness and contrast are the optimal settings to improve ASCII quality
    # Quality should be the same with ProcessingPipeline and ProcessingPipelineAscii
    # pipeline_ascii = ProcessingPipelineAscii(brightness=100, contrast=3.0)
    pipeline_ascii = ProcessingPipeline(brightness=100, contrast=3.0)

    pipeline_ascii.converter = converter
    pipeline_ascii.img_width = img_width
    pipeline_ascii.img_height = img_height

    _ = pipeline_ascii.run(image)
    ascii_img = pipeline_ascii.ascii

    return _        # DEBUG - we are returning the full color image for now

def init_converter(charset: Charset, converter_class=None):
    """Initialize the ASCII converter with C64 settings.
    """
    # Initialize the ML converter trained with C64 charset

    try:
        converter = None

        # Initialize manual converter(s)
        if not converter_class:
            # converter = ShiftingAsciiConverter(charset) # medium accuracy
            converter_class = 'FeatureAsciiConverter' # high accuracy

        if converter_class not in sys.modules[__name__].__dict__:
            raise NameError()

        class_def = sys.modules[__name__].__dict__[converter_class]

        if converter_class == 'NeuralAsciiConverterPytorch':
            converter = class_def(
                charset=charset,
                model_filename=MODEL_FILENAME,
                model_charset=MODEL_CHARSET,
                charsize=[charset.char_width, charset.char_height],
                num_labels=len(charset.chars)
            )
        else:
            converter = class_def(charset)

    except NameError as e:
        raise Exception(f"Unable to find converter: {converter_class}.\n{e}")

    return converter


if __name__ == '__main__':
    """ handle command line arguments using argparse"""
    """ Defaults:
    out_width, out_height = 384, 384        # 48x48 characters of 8x8
    """

    import argparse
    parser = argparse.ArgumentParser(description='Convert an image to ASCII art.')
    parser.add_argument(
        'input',
        type=str,
        help='Path to the input image file (PNG format).')
    parser.add_argument(
        '--width',
        type=int,
        help='Width of the output image.',
        default=384)
    parser.add_argument(
        '--height',
        type=int,
        help='Height of the output image.',
        default=384)
    parser.add_argument(
        '--charset',
        type=str,
        help='Name of the charset image to use (incl. file extension).',
        default=CHARSET_NAME)
    parser.add_argument(
        '--converter',
        type=str,
        help='Name of the ASCII converter to use.',
        default='FeatureAsciiConverter')

    args = parser.parse_args()

    input_img = args.input

    # Remove .PNG extension from file name
    output_img = input_img.split('.')[:-1][0]
    output_img = output_img + '_ascii.png'

    width = args.width
    height = args.height
    charset_name = args.charset
    converter = args.converter

    # Load a charset
    charset = Charset()
    charset.load(charset_name)

    output_img_bin = convert_image(input_img, charset, img_width=width, img_height=height, converter_class=converter)

    # Save the final image
    cv.imwrite(output_img, output_img_bin)
    print(f'-- Output image: {output_img} --')


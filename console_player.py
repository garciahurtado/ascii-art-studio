import sys

from ascii import NeuralAsciiConverterPytorch
from charset import Charset
from cpeg.encoder import Encoder
from cvtools.processing_pipeline import ProcessingPipeline
import cv2 as cv

def main(image):
    char_width, char_height = 8, 16
    charset = load_charset(char_width, char_height)
    convert_image(charset, image)
    pass

def load_charset(char_width, char_height):

    charset = Charset(char_width, char_height)
    charset_name = 'ubuntu-mono_8x16.png'
    charset.load(charset_name, invert=False)

    return charset


def run_convert_to_ascii(source_image, pipeline, charset):
    height, width = source_image.shape[0], source_image.shape[1]
    height = height - (height % charset.char_height)
    width = width - (width % charset.char_width)
    final = pipeline.run(source_image)

    return print_char_map(pipeline.converter.match_char_map)

def print_char_map(char_map):
    for row in char_map:
        for block in row:
            char = block.character
            fg_color, bg_color = block.fg_color, block.bg_color
            fg_color, bg_color = rgb_to_ansi(fg_color, bg_color)
            hex_value = char.code[2:]
            code_point = int(hex_value, 16)
            unicode_char = chr(code_point)
            print(f"{fg_color}{bg_color}{unicode_char}"+"\033[0m", end='')

            #bytes = bytearray.fromhex(char.code.removeprefix('0x'))
            #sys.stdout.buffer.write(bytes)

        print('')

def convert_image(charset, image_path):
    width, height = 640, 368  # For video
    num_chars = 734
    char_height, char_width = charset.char_height, charset.char_width

    converter = NeuralAsciiConverterPytorch(charset, 'ubuntu_mono-Mar24_01-42-34', 'ubuntu_mono', [8, 16],
                                            num_labels=num_chars)

    pipeline = get_pipeline(converter, height, width, char_height, char_width)
    source_image = cv.imread(image_path)

    run_convert_to_ascii(source_image, pipeline, charset)

def get_pipeline(converter, height, width, char_height, char_width):
    pipeline = ProcessingPipeline()
    pipeline.converter = converter
    pipeline.img_width = width
    pipeline.img_height = height
    pipeline.char_height = char_height
    pipeline.char_width = char_width

    return pipeline

def rgb_to_ansi(bg_color, fg_color):

    # Calculate the closest ANSI color codes for background and foreground
    bg_ansi = '\033[48;2;{};{};{}m'.format(*bg_color)
    fg_ansi = '\033[38;2;{};{};{}m'.format(*fg_color)

    # Return the ANSI color control codes
    return bg_ansi, fg_ansi

if __name__ == '__main__':
    image = 'resources/images/darth-vader-bust.png'
    main(image)

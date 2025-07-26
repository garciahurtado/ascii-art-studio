import cv2 as cv

from ascii import NeuralAsciiConverterPytorch
from charset import Charset
from cpeg.encoder import Encoder
from cvtools.processing_pipeline_ascii import ProcessingPipelineAscii
from cvtools.processing_pipeline_color import ProcessingPipelineColor


def convert_image(image_path, output_path, charset, img_height=None, img_width=None):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    encoder = create_encoder(output_path, charset, (img_width, img_height))

    converter = NeuralAsciiConverterPytorch(charset, 'ubuntu_mono-Mar24_01-42-34', 'ubuntu_mono', [8, 16],
                                            num_labels=len(charset))

    pipeline_ascii = ProcessingPipelineAscii()
    pipeline_ascii.converter = converter
    pipeline_ascii.img_width = img_width
    pipeline_ascii.img_height = img_height

    pipeline_color = ProcessingPipelineColor()
    pipeline_color.img_width = img_width
    pipeline_color.img_height = img_height

    max_palette_frames = 30  # Change palette after n frames
    palette_frame_count = 0  # Keep track of last palette change

    [ascii, contrast, color] = pipeline_ascii.run(image)
    charmap = pipeline_ascii.converter.match_char_map

    print(f"ASCII charmap generated: {charmap.shape[0]} x {charmap.shape[1]}")

    color_idx, palette = pipeline_color.extract_palette(image, num_colors=256)
    pipeline_color.palette = palette

    # Export the palette to binary before exporting the frames that use it
    encoder.export_binary_palette(pipeline_color.palette)

    # Export to binary file
    middle = charmap.shape[0] * charmap.shape[1]
    charmap = encoder.embed_colors(charmap, color_idx[:middle], color_idx[middle:])

    encoder.export_binary_frame(charmap, is_full=True)
    #encoder.export_ascii_data(charmap, pipeline_ascii.fg_colors, pipeline_ascii.bg_colors)

    return [charmap, ascii, contrast, color]


def create_encoder(binary_output_file, charset, resolution):
    encoder = Encoder(resolution, (charset.char_width, charset.char_height), binary_output_file)
    encoder.export_binary_header(1, 0, 30, charset.filename)
    return encoder


if __name__ == '__main__':
    image_path = 'resources/images/garcia-retrato.png'
    output_path = 'resources/cpeg/garcia-retrato.cpeg'
    out_width, out_height = 640, 368

    # Load a charset
    char_width, char_height = 8, 16
    charset = Charset(char_width, char_height)
    charset_name = 'ubuntu-mono_8x16.png'
    charset.load(charset_name, invert=False)
    charmap = []
    num_chars = 734

    convert_image(image_path, output_path, charset, img_width=out_width, img_height=out_height)

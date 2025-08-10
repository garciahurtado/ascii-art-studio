import argparse
import collections
import os
import sys
import time
import cv2 as cv
import numpy as np
import threading

from color import palette
from cpeg.encoder import Encoder
from cvtools.processing_pipeline import ProcessingPipeline
from cvtools.processing_pipeline_ascii import ProcessingPipelineAscii
from cvtools.processing_pipeline_color import ProcessingPipelineColor
from ui import looper
from ui.block_grid_window import BlockGridWindow
from ascii.neural_ascii_converter_pytorch import NeuralAsciiConverterPytorch
from charset import Charset
from color.palette import Palette
from cvtools import contrast_filters as filters, size_tools as tools, color_filters as colors

from ui.video_player import VideoPlayer
from const import PROJECT_ROOT

print("OpenCV Version: ")
print(cv.__version__)

screenshot_dir = 'resources/screenshots/'

show_grid = False
ascii_scale = 2
resize_threshold = 64

mouse_x, mouse_y = 0, 0
selected_block = None

FRAME_FULL = 1
FRAME_DIFF = 2
FRAME_PALETTE = 3

dilate_kernel = np.array(([0, 1, 0], [1, 2, 1], [0, 1, 0]), np.uint8)
is_full = True  # First frame is always a full frame

PALETTE_CHANGE_SIGNAL = 77

last_frame_time = 0
last_palette = None
binary_output_file = 'web/static/charpeg/video_stream.cpeg'

# Show controls
def show_controls():
    cv.namedWindow('controls')
    cv.createTrackbar('grid', 'controls', show_grid, 1, set_grid)
    cv.createTrackbar('ascii_sc', 'controls', 3, 3, set_ascii_scale)
    cv.createTrackbar('resizeth', 'controls', 64, 256, set_resize_threshold)
    cv.resizeWindow('controls', 400, 100)

@looper.fps(30)
def run_block_contrast(ui, pipeline, width, height, video_player=None, layer_window=None):
    global total_chars
    global ascii_scale
    global show_grid
    global now

    ## VIDEO
    original = video_player.get_frame()

    # MoviePy uses RGB, so convert it to BGR
    original = cv.cvtColor(original, cv.COLOR_RGB2BGR)

    # ## IMAGE
    #
    # img_path = img
    # ui.image_path = img_path
    # original = cv.imread(img_path)

    ## PROCESSING ##
    final = pipeline.run(original)
    # _, final = colors.palettize(final, palette)

    # Double up the final image
    final = cv.resize(final, (width * 4, height * 4), interpolation=cv.INTER_NEAREST)

    chars = converter.match_char_map
    num_chars = converter.count_used_chars()

    ui.ui_text['# chars: '] = f'{num_chars}/{num_chars}'

    # if ui.show_layer is not None:
    #     final = pipeline.__getattribute__(ui.show_layer)
    #     final = cv.resize(final, (width * 2, height * 2), interpolation=cv.INTER_NEAREST)
    #     if(len(final.shape) == 2):
    #         final = cv.cvtColor(final, cv.COLOR_GRAY2RGB)

    # write out data file with ASCII version of image

    # export_ascii_data(chars, pipeline.fg_colors, pipeline.bg_colors)

    # if selected_block and char_window:
    #     ui.select_block(selected_block)
    #     char_window.select_block(selected_block)
    ui.show(final)

    # char_window.show(pipeline.contrast_img)
    # char_window.show_char_matches(pipeline.contrast_img)
    if layer_window:
        show_layer_control(layer_window, ui)

    return ui.get_key()

@looper.fps(30)
def run_convert_to_png(ui, charset, converter, image_path):
    image = cv.imread(image_path)
    height, width = image.shape[0], image.shape[1]
    height = height - (height % charset.char_height)
    width = width - (width % charset.char_width)
    pipeline = get_pipeline(converter, height, width, charset.char_height, charset.char_width)
    final = pipeline.run(image)
    ui.show(final)
    return ui.get_key()

@looper.fps(60)
def run_webcam(ui, charset, converter):
    width, height = 960, 720  # For Logitech webcam video
    # WEBCAM ----
    video = cv.VideoCapture(0, cv.CAP_DSHOW) # webcam
    video.set(cv.CAP_PROP_BUFFERSIZE, 3)
    video.set(cv.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    _, original = video.read()

    # Resize
    width, height = int(width / 2), int(height / 2)
    height = height + 8
    pipeline = get_pipeline(converter, height, width, charset.char_height, charset.char_width)

    #resized = cv.resize(original, (width, height))

    final = pipeline.run(original)
    ui.show(final)
    return ui.get_key()

def run_all_frames_setup(ui, video, charset):
    # max_frames = 30
    max_frames = None

    video.pause()
    ascii_frame_buffer = collections.deque()
    render_frame_buffer = collections.deque()

    encoder = Encoder(video.resolution, (charset.char_width, charset.char_height), binary_output_file)

    if (max_frames):
        num_frames = max_frames
        duration = num_frames / video.fps
    else:
        num_frames = video.num_frames
        duration = video.duration

    encoder.export_binary_header(num_frames, duration, video.fps, charset_name)

    ascii_thread = threading.Thread(
        target=run_all_frames_ascii_thread,
        args=[ascii_frame_buffer, max_frames],
        daemon=True)
    ascii_thread.start()

    all_frames_thread = threading.Thread(
        target=run_all_frames_color_thread,
        args=[ascii_frame_buffer, render_frame_buffer, encoder, max_frames],
        daemon=True)
    all_frames_thread.start()

    run_all_frames_render(ui, render_frame_buffer)

    all_frames_thread.join()
    ascii_thread.join()


def run_all_frames_ascii_thread(frame_buffer, video, max_frames=None):
    pipeline_ascii = ProcessingPipelineAscii()
    pipeline_ascii.converter = converter
    pipeline_ascii.img_width = width
    pipeline_ascii.img_height = height

    max_palette_frames = 30  # Change palette after n frames
    palette_frame_count = 0  # Keep track of last palette change
    frame_num = 0

    for frame in video.get_next_frame():
        time.sleep(0.001)  # Let other threads work

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        ascii = pipeline_ascii.run(frame)
        contrast = pipeline_ascii.contrast_img
        color = pipeline_ascii.color
        charmap = pipeline_ascii.converter.match_char_map

        print("ASCII Frame decoded")

        if (palette_frame_count > max_palette_frames):
            frame_buffer.append(PALETTE_CHANGE_SIGNAL)
            palette_frame_count = 0

        frame_buffer.append([charmap, ascii, contrast, color])
        palette_frame_count += 1
        frame_num += 1

        if max_frames and frame_num >= max_frames:
            break

    print("END ASCII THREAD")


def run_all_frames_color_thread(ascii_frame_buffer, render_frame_buffer, encoder, pipeline_color=None, max_frames=None):
    global PALETTE_CHANGE_SIGNAL
    global last_char_map

    color_frame_buffer = []
    palette = None
    num_colors = 128
    frame_num = 0

    while (True):
        time.sleep(0.001)  # Let other threads work

        if (len(ascii_frame_buffer) > 0):
            message = ascii_frame_buffer.popleft()

            if message == PALETTE_CHANGE_SIGNAL:
                print("Palette change")

                # ASCII frame buffer arrays: [charmap, ascii_image, fg_colors, bg_colors]
                all_frames = [np.vstack([frame[2], frame[3]]) for frame in color_frame_buffer]
                all_frames = np.vstack(all_frames)
                color_idx, palette = pipeline_color.extract_palette(all_frames, num_colors=num_colors)
                pipeline_color.palette = palette

                # Export the palette to binary before exporting the frames that use it
                encoder.export_binary_palette(pipeline_color.palette)

                # Store the color indices alongside each frame, to make it easier to palettize them
                frame_charmap = color_frame_buffer[0][0]
                size = frame_charmap.shape[0] * frame_charmap.shape[1] * 2  # One for each color

                for i in range(0, len(color_frame_buffer)):
                    frame_color_idx = color_idx[i * size: (i * size) + size]
                    color_frame_buffer[i].append(frame_color_idx)
                    char_map = color_frame_buffer[i][0]

                    # Export to CPEG binary file
                    middle = char_map.shape[0] * char_map.shape[1]
                    char_map = encoder.embed_colors(char_map, frame_color_idx[:middle], frame_color_idx[middle:])

                    encoder.export_full_or_diff_frame(char_map, last_char_map)
                    last_char_map = char_map.copy()

            else:
                [charmap, ascii, contrast, color] = message

                start = time.process_time()
                pipeline_color.ascii = ascii
                pipeline_color.color = color
                pipeline_color.run(contrast)
                color_frame_buffer.append([charmap, ascii, pipeline_color.fg_colors, pipeline_color.bg_colors])

                elapsed_time = (time.process_time() - start) * 1000

                print(f"Color image rendered in {elapsed_time:.2f}ms... {len(ascii_frame_buffer)} left in queue")

            if palette is not None:
                for [charmap, frame_ascii, fg_colors, bg_colors, frame_color_idx] in color_frame_buffer:
                    time.sleep(0.000001)

                    # pipeline_color.palettize(fg_colors, bg_colors, frame_color_idx)
                    final_frame = pipeline_color.compose(frame_ascii)
                    final_frame = pipeline_color.resize(final_frame)

                    render_frame_buffer.append([final_frame, pipeline_color.palette.colors])
                    frame_num += 1

                palette = None
                color_frame_buffer = []

            if max_frames and frame_num >= max_frames:
                break

    print("END COLOR THREAD")


@looper.fps(12000)
def run_all_frames_render(ui, render_frame_buffer, fps=4):
    frame_time_ms = 1000 / fps
    global last_frame_time
    global last_palette
    is_palette_shown = False

    time_since_last_frame = round(time.time() * 1000) - last_frame_time

    if len(render_frame_buffer) > 0 and (time_since_last_frame > frame_time_ms):
        [final_image, palette] = render_frame_buffer.popleft()

        if is_palette_shown and palette is not None:
            final_image = show_palette(400, 5, palette, final_image)
            last_palette = palette

        ui.show(final_image)
        last_frame_time = round(time.time() * 1000)
        time.sleep(0.1)

    return ui.get_key()


@looper.fps(30)
def run_orig():
    ui.ui_text['PALETTE'] = len(palette.colors)

    # ret, original = video.read()
    original = cv.imread('resources/test/joker.png')
    original = cv.resize(original, (width, height), interpolation=cv.INTER_NEAREST)  # Knock down the res for testing

    colorized = original.copy()
    colorized = cv.convertScaleAbs(colorized, 1, 1.4)
    colorized = colors.pixelize(colorized)
    colorized = colors.brightness_saturation(colorized, 0.9, 1.6)
    colors1, colorized = colors.palettize(colorized, palette)

    colorized_dark = cv.convertScaleAbs(original.copy(), 0.8, 1.2)
    colorized_dark = colors.pixelize(colorized_dark)
    colorized_dark = colors.brightness_saturation(colorized_dark, 0.5, 1.8)
    colors2, colorized_dark = colors.palettize(colorized_dark, palette)

    ui.ui_text['COL USED'] = len(set(colors1 + colors2))

    inverted = filters.invert_mask(original.copy())

    neg_inverted = cv.bitwise_not(inverted.copy())

    # Edge detection
    edges = original.copy()
    edges = cv.medianBlur(edges, 3)
    edges = cv.Canny(edges, 60, 200)
    ascii = cv.dilate(edges, dilate_kernel)

    # Convert to 2-bit and ASCII-ize
    res, ascii = cv.threshold(ascii, 100, 255, cv.THRESH_BINARY_INV)
    _, ascii = converter.convert_image(ascii)
    ascii = cv.cvtColor(ascii, cv.COLOR_GRAY2RGB)

    # Turn it green for "matrix style"
    # img[:, :, 0] = 0
    # img[:, :, 2] = 0

    ascii_inverted = ascii.copy()

    ascii = cv.bitwise_xor(ascii, inverted)
    ascii_dark = cv.bitwise_xor(ascii_inverted, neg_inverted)

    blended1 = cv.bitwise_and(ascii, colorized)
    blended2 = cv.bitwise_and(ascii_dark, colorized_dark)
    blended_final = cv.bitwise_or(blended2, blended1)

    # Show final image
    final = cv.resize(blended_final, (blended_final.shape[1] * 2, blended_final.shape[0] * 2),
                      interpolation=cv.INTER_NEAREST)
    if show_grid:
        ui.show_grid = True

    ui.show(final)
    ui.take_screenshot()

    return ui.get_key()


def get_pipeline(converter, width, height, char_height, char_width):
    pipeline = ProcessingPipeline(brightness=100, contrast=3)
    pipeline.converter = converter
    pipeline.img_width = width
    pipeline.img_height = height
    pipeline.char_height = char_height
    pipeline.char_width = char_width
    pipeline.palette = palette

    return pipeline


def run_player(ui, charset, converter, video_file, out_width, out_height):
    # Load a palette
    palette = Palette(char_width=charset.char_width, char_height=charset.char_height)
    palette.load('atari-st.png')

    # cv.imshow('charset_res_map', charset.show_low_res_maps())

    #show_controls()
    #show_video_controls(ui)
    layer_window = None

    # cProfile.run('run_block_contrast(ui, char_window, layer_window)', sort='tottime')
    # run_all_frames_setup(ui)

    pipeline = get_pipeline(converter, out_width * 2, out_height * 2, char_height, char_width)

    player = VideoPlayer(video_file, resolution=(out_width, out_height), zoom=2)
    player.play(sound=True)

    """ 
    Other functionality provided by the script:
        run_convert_to_png(ui, charset, converter, 'resources/images/garcia-retrato.png')
        run_webcam(ui, charset, converter)
    """

    run_block_contrast(ui, pipeline, width, height, player, layer_window=layer_window)
    # 'resources/video/Star Wars - Opening Scene.mp4'
    # 'resources/video/Akira - bike scene.mp4'


def get_ui(converter):
    ui = BlockGridWindow('output')
    ui.converter = converter
    ui.show_fps = False
    ui.show_layer = None
    cv.moveWindow('output', 550, 20)

    # char_window = CharPickerWindow('characters')
    # char_window.converter = converter
    # char_window.ui_window = ui
    # cv.moveWindow('characters', 10, 750)
    char_window = None

    ui.char_picker_window = char_window
    return ui


def get_converter(charset, model_type, model_name):
    char_width = charset.char_width
    char_height = charset.char_height
    num_chars = len(charset.chars)
    converter = NeuralAsciiConverterPytorch(charset, model_name, model_type, [char_width, char_height],
                                            num_labels=num_chars)
    return converter


if __name__ == "__main__":
    """ 
        Other functionality provided by the script:
            run_convert_to_png(ui, charset, converter, 'resources/images/garcia-retrato.png')
            run_webcam(ui, charset, converter)
        """
    parser = argparse.ArgumentParser()

    parser.add_argument("--char_width", type=int, default=8)
    parser.add_argument("--char_height", type=int, default=8)
    parser.add_argument("--convert_to_png", action='store_true', default=False)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--input", type=str, default=None, required="--convert_to_png" in sys.argv)
    parser.add_argument("--webcam", action='store_true', default=False)
    args = parser.parse_args()

    # All actions require a charset
    char_width, char_height = 8, 8
    charset = Charset(char_width, char_height)
    # charset.load('c64.png')
    charset.load('amstrad-cpc.png')

    # model_type = 'ascii_c64'
    # model_name = 'ascii_c64'

    model_type = 'AsciiAmstradCPC'
    model_name = 'AsciiAmstradCPC-Mar06_23-14-37'
    video_dir = os.path.join('resources', 'source_video')

    if args.convert_to_png:
        run_convert_to_png(args.char_width, args.char_height, args.input, args.output_dir)
    elif args.webcam:
        converter = get_converter(charset, model_type, model_name)
        ui = get_ui(converter)
        run_webcam(ui, charset, converter)
    else:
        # Run desktop video player
        # width, height = 496, 368 # For images
        # width, height = 992, 736 # For images
        # width, height = 640, 368  # For video
        width, height = 320, 184  # For video . Halved because we apply x2 zoom later
        # width, height = 640, 272 # Luke Darth video

        converter = get_converter(charset, model_type, model_name)
        # converter.set_region((0, 5),(12, 25)) # Death star
        # converter.set_region((35, 11),(60, 26)) # Star Wars

        ui = get_ui(converter)
        video_file = os.path.join(video_dir, 'Star Wars - Opening Scene.mp4')
        run_player(ui, charset, converter, video_file, width, height)

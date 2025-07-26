import math
import numpy as np
import cv2 as cv
import PySimpleGUI as gui

def create_layer_control():
    button_size = (30, 3)

    layout = [[gui.Text('Show Layer')],
              [gui.Button('Original', size=button_size)],
              [gui.Button('Otsu Contrast', size=button_size)],
              [gui.Button('ASCII', size=button_size)],
              [gui.Button('Grid On', size=button_size)],
              [gui.Button('Grid Off', size=button_size)],
              [gui.Button('Final', size=button_size)]
              ]
    window = gui.Window('Layer Control', layout, location=(100, 100))
    return window


def show_layer_control(layer_window, ui):
    # Layer control GUI
    event, values = layer_window.read(timeout=300)  # Use timeout so we don't block waiting for layer window input
    if event == 'Original':  # if user closes window or clicks cancel
        ui.show_layer = 'original'
    elif event == 'Otsu Contrast':
        ui.show_layer = 'contrast_img'
    elif event == 'ASCII':
        ui.show_layer = 'ascii'
    elif event == 'Final':
        ui.show_layer = None
    elif event == 'Grid On':
        ui.show_grid = True
    elif event == 'Grid Off':
        ui.show_grid = False


def set_diff_empty_threshold(value, charset):
    charset.diff_empty_threshold = value


def set_diff_full_threshold(value, charset):
    charset.diff_full_threshold = value


def set_diff_match_min_threshold(value, charset):
    charset.diff_match_min_threshold = value


def set_diff_match_max_threshold(value, charset):
    charset.diff_match_max_threshold = value


def set_ascii_scale(value):
    global ascii_scale
    ascii_scale = value


def set_resize_threshold(value):
    global resize_threshold
    resize_threshold = value


def set_grid(value):
    global show_grid
    show_grid = value


def show_palette(start_x, start_y, colors, orig):
    total_rows = 10
    total_cols = 50
    palette_img = np.zeros([total_rows, total_cols, 3], dtype=np.uint8)

    for i, color in enumerate(colors):
        col = i % total_cols
        row = math.floor(i / total_cols)
        palette_img[row, col] = color

    palette_img = cv.resize(palette_img, (total_cols * 8, total_rows * 8), interpolation=cv.INTER_NEAREST)
    width, height = palette_img.shape[1], palette_img.shape[0]
    orig[start_y:start_y + height, start_x:start_x + width] = palette_img
    return orig


def mouse_click(event, x, y, flags, param, char_width, char_height):
    global mouse_x, mouse_y
    global selected_block

    if event == cv.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = math.floor(x / 2), math.floor(y / 2)
        mouse_x = mouse_x - (mouse_x % char_width)
        mouse_y = mouse_y - (mouse_y % char_width)

        sel_col = int(mouse_x / char_width)
        sel_row = int(mouse_y / char_height)
        selected_block = (sel_row, sel_col)

def show_video_controls(ui):
    win = cv.namedWindow(ui.window_name)
    # Create buttons
    button_width = 100
    button_height = 50
    button_color = (200, 200, 200)
    text_color = (0, 0, 0)
    button_gap = 10

    # Create a blank image for buttons
    buttons = np.zeros((button_height, win.frame_width, 3), np.uint8)

    # Create rewind button
    cv.rectangle(buttons, (button_gap, 0), (button_gap + button_width, button_height), button_color, -1)
    cv.putText(buttons, 'Rewind', (button_gap + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Create pause/play button
    cv.rectangle(buttons, (button_gap * 2 + button_width, 0), (button_gap * 2 + button_width * 2, button_height),
                  button_color, -1)
    cv.putText(buttons, 'Pause/Play', (button_gap * 2 + button_width + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                text_color, 2)

    # Create end button
    cv.rectangle(buttons, (button_gap * 3 + button_width * 2, 0), (button_gap * 3 + button_width * 3, button_height),
                  button_color, -1)
    cv.putText(buttons, 'End', (button_gap * 3 + button_width * 2 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color,
                2)

    # Initialize variables
    current_frame = 0
    playing = True

    # while True:
    #     if playing:
    #         ret, frame = video.read()
    #         if not ret:
    #             break
    #         current_frame += 1
    #     else:
    #         frame = np.zeros((frame_height, frame_width, 3), np.uint8)
    #
    #     # Combine the video frame and buttons
    #     frame_with_buttons = np.vstack((frame, buttons))
    #
    #     cv.imshow('Video Player', frame_with_buttons)
    #
    #     key = cv.waitKey(int(1000 / fps))
    #     if key == ord('q'):
    #         break
    #     elif key == ord('r'):
    #         video.set(cv.CAP_PROP_POS_FRAMES, 0)
    #         current_frame = 0
    #     elif key == ord('p'):
    #         playing = not playing
    #     elif key == ord('e'):
    #         current_frame = total_frames - 1
    #         video.set(cv.CAP_PROP_POS_FRAMES, current_frame)
    #         playing = False
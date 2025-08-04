import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def extract_colors_from_mask(color_img, mask, contrast_mask, block_size, invert=False):
    if invert:
        mask = cv.bitwise_not(mask)
        contrast_mask = cv.bitwise_not(contrast_mask)

    width, height = color_img.shape[1], color_img.shape[0]
    block_width, block_height = block_size
    num_rows = height / block_height
    num_cols = width / block_width

    if(num_rows != int(num_rows) or num_cols != int(num_cols)):
        raise ValueError(f"The image ({width}x{height}) is not divisible by the character size ({block_width}x{block_height})")

    num_rows = int(num_rows)
    num_cols = int(num_cols)

    out_width, out_height = int(width / block_width), int(height / block_height)
    out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    for row in range(0, num_rows):
        y = row * block_height

        for col in range(0, num_cols):
            x = col * block_width

            mask_block = mask[y : y + block_height, x : x + block_width]
            contrast_block = contrast_mask[y : y + block_height, x : x + block_width]
            color_block = color_img[y : y + block_height, x : x + block_width]

            color = extract_color_from_block(color_block, mask_block, contrast_block)
            out_img[row, col] = color


    return out_img



def extract_colors_from_char_mask(img, characters, invert=False):
    #@TODO   : deprecate
    # Valid scales: 1/4, 1/2 or 1/1
    scale = 1/8

    width, height = img.shape[1], img.shape[0]
    # img = pixelize(img, scale=1 / 4)

    # reduce the sampling image resolution
    # img = cv.resize(img, (int(width * scale), int(height * scale)))

    # We halve the scale of the output image because the scale for sampling foreground / background
    # colors is twice as big as the scale at which we store the extracted color (1 color per 8x8 block)
    out_width, out_height = int(width / 8), int(height / 8)
    out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    for i, row in enumerate(characters):
        for j, character in enumerate(row):
            char_size = int(8 * scale)

            if scale == 1:
                character = character
            else:
                character = character.low_res[char_size]

            char_img = character.img

            if invert:
                char_img = cv.bitwise_not(char_img)

            x = j * int(8 * scale)
            y = i * int(8 * scale)
            color = extract_color_from_block(
                img[y : y + char_size, x : x + char_size], char_img
            )

            out_img[i, j] = color

    # Now, resize it back up
    out_img = cv.resize(out_img, (width, height), interpolation=cv.INTER_NEAREST)

    return out_img


def extract_color_from_block(img, mask, contrast_mask):
    """ Assumes BGR color format for the img"""
    avg_ascii = cv.mean(img, mask=mask)
    avg_contrast = cv.mean(img, mask=contrast_mask)

    color1 = (avg_ascii[0], avg_ascii[1], avg_ascii[2])
    color2 = (avg_contrast[0], avg_contrast[1], avg_contrast[2])

    # average the two colors
    final = (int((color1[0] + color2[0]) / 2),
             int((color1[1] + color2[1]) / 2),
             int((color1[2] + color2[2]) / 2))

    return final


def palettize(img, palette, scale=1, color_idx=None):
    """
    Convert the colors of the passed image to the closest colors available in the palette provided

    :param img: An np.array of dims [height,width,3]
    :param palette: A 1D array of color triplets
    :param scale: If less than one, will downsample the source image before converting it to the palette.
    :return:
    """
    # img = cv.cvtColor(img, cv.COLOR_RGB2LAB)

    if color_idx is not None:
        color_idx = color_idx.reshape([img.shape[0], img.shape[1]])

        for y, row in enumerate(img):
            for x, col in enumerate(row):
                idx = color_idx[y,x]
                img[y,x] = palette.colors[idx]

        color_len = color_idx.shape[0] * color_idx.shape[1]
        return color_idx.reshape([color_len]), img

    # Make it small
    if scale < 1:
        orig_width, orig_height = img.shape[1], img.shape[0]
        height, width = int(orig_height * scale), int(orig_width * scale)
        img = cv.resize(img, (width, height), 0, 0, cv.INTER_NEAREST)

    width = img.shape[1]
    height = img.shape[0]

    colors = img.reshape([height * width,3])
    labels = palette.clusters.predict(colors)
    indexed_colors = [palette.colors[idx] for idx in labels]
    indexed_colors = np.asarray(indexed_colors, dtype=np.uint8)
    img = indexed_colors.reshape([height, width, 3])

    # img = cv.cvtColor(img, cv.COLOR_LAB2RGB)

    # Scale it back up
    if scale < 1:
        img = cv.resize(img, (orig_width, orig_height), interpolation=cv.INTER_NEAREST)

    return labels, img

def pixelize(img, color_reduce=8, scale=1 / 8):
    """
    Shrinks the image, reduces total number of colors and scales it back up, in order to provide a blocky, pixelated,
    color-limited result. Reducing the number of colors is useful in order to improve performance of a future "palettize"
    operation.
    """
    orig_height = img.shape[0]
    orig_width = img.shape[1]

    img = cv.blur(img, (5, 5), 2)

    small_height = int(img.shape[0] * scale)
    small_width = int(img.shape[1] * scale)

    # Shrink it
    # After multiple tests, Bicubic interpolation gives the best results here
    img = cv.resize(img, (small_width, small_height), interpolation=cv.INTER_CUBIC)

    # Color reduction
    #  TODO: improve. Do not reduce pure white or pure black colors
    div = color_reduce
    img = img // div * div + div // 2

    # Scale it back up
    img = cv.resize(img, (orig_width, orig_height), interpolation=cv.INTER_NEAREST)

    return img


def brightness_saturation(img, brightness=1, saturation=1):
    """Increase or decrease the brightness and saturation of an image in order to make it "pop"
    (or make a duller / darker version)"""

    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsvImg = np.int16(hsvImg)

    # increase saturation
    hsvImg[..., 1] = hsvImg[..., 1] * saturation

    # multiple by a factor of less than 1 to reduce the brightness
    hsvImg[..., 2] = hsvImg[..., 2] * brightness

    # Clip the results to avoid out of range issues
    hsvImg = np.clip(hsvImg, 0, 255)
    hsvImg = np.uint8(hsvImg)

    img = cv.cvtColor(hsvImg, cv.COLOR_HSV2BGR)
    return img


def brightness_contrast(input_img, brightness=0, contrast=0):
    """ @ref: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def quantize_img(img, bitdepth=[3,4,4]):
    ''' We quantize every color in the image at once by creating a special image which
    works as a mask to remove the least significant bits of each color. We apply
    this mask with bitwise AND to every pixel in the NumPy array at once. '''

    height = img.shape[0]
    width = img.shape[1]

    # Retain downsampled values for each of Red, Green and Blue components
    masks = []
    for i, bits in enumerate(bitdepth):
        # ie: 3 bits = 11100000
        mask = int(('1'*bits) + ('0'*(8-bits)), 2)
        masks.append(mask)

    color_mask = np.full([height, width, 3], masks, np.uint8)
    img = np.bitwise_and(img, color_mask)

    return img

def quantize_color_division(pixel, factor=8):
    # @DEPRECATED
    # The higher the factor, the lower number of total colors

    r, g, b = pixel[0], pixel[1], pixel[2]

    r = math.floor(r / factor)
    r = int(r * factor)

    g = math.floor(g / factor)
    g = int(g * factor)

    b = math.floor(b / factor)
    b = int(b * factor)

    return (r, g, b)


def show_histogram(img):
    # create a mask that ignores pure black and pure white pixels
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    mask = (v == 0) + (v == 100) + (s == 0)
    mask = np.logical_not(mask)
    mask = np.array(mask, dtype=np.uint8)

    # Split the B, G and R channels
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    channels = cv.split(img)
    colors = ("b", "g", "r")

    fig = plt.figure(0, figsize=(15, 8))
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")

    for (channel, color) in zip(channels, colors):
        hist = cv.calcHist([channel], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.show()


def extract_palette(img):
    """
    Return palette of color triplets in descending order of frequency
    """
    # print(f"Extracting palette from image: {img.shape[1]} x {img.shape[0]}")

    arr = np.asarray(img)
    palette, index = np.unique(asvoid(arr).ravel(), return_inverse=True)
    palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])

    return palette


def asvoid(arr):
    """View the array as dtype np.void (bytes)
    This collapses ND-arrays to 1D-arrays, so you can perform 1D operations on them.
    http://stackoverflow.com/a/16216866/190597 (Jaime)
    http://stackoverflow.com/a/16840350/190597 (Jaime)
    Warning:
    # >>> asvoid([-0.]) == asvoid([0.])
    array([False], dtype=bool)
    """
    ret_arr = np.ascontiguousarray(arr)
    return ret_arr.view(np.dtype((np.void, ret_arr.dtype.itemsize * ret_arr.shape[-1])))


def totuple(a):
    '''
    @ref: https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
    '''
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


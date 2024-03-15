import math
import random

import cv2 as cv
import numpy as np
import cvtools.size_tools as tools

"""
Series of utility functions which perform various filters on images themed around contrast and 2 bit images
"""


def block_contrast(input_image, block_size, invert=False):
    """WARNING: This function modifies the input image (for efficiency), so make sure you make a copy of it if you don't want that"""
    # rand_inv = Whether to randomly invert half of the blocks processed,
    # in order to provide a more balanced output for ML training

    # @TODO: explore this threshold mechanism some other time
    # output = cv.adaptiveThreshold(input_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 10)
    # return output

    block_width, block_height = block_size[1], block_size[0]
    img_width, img_height = input_image.shape[1], input_image.shape[0]
    num_cols, num_rows = math.floor(img_width / block_width), math.floor(img_height / block_height)

    blocks = tools.as_blocks(input_image, (block_height, block_width), flatten=True)

    for block_num, block in enumerate(blocks):
        col_num = block_num % num_cols
        row_num = math.floor(block_num / num_cols)
        x = col_num * block_width
        y = row_num * block_height

        contrast_block = image_contrast(block)
        if(invert):
            contrast_block = np.invert(contrast_block)

        input_image[y:y+block_height, x:x+block_width] = contrast_block

    return input_image


def image_contrast(img):
    """Converts image to 2bit high-contrast using the OTSU thresholding function. Must be a grayscale image"""

    _, contrast = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    return contrast


def block_mask(source):
    """Very simple mask which will only return black blocks where the entire 8x8 block is black.
    Input image must be in grayscale format"""

    orig_height, orig_width = source.shape[0], source.shape[1]

    # Scale down
    new_width = int(source.shape[1] / 8)
    new_height = int(source.shape[0] / 8)
    source = cv.resize(source, (new_width, new_height), interpolation=cv.INTER_AREA)
    _, mask = cv.threshold(source, 1, 255, cv.THRESH_BINARY)

    # Scale back up
    mask = cv.resize(mask, (orig_width, orig_height), interpolation=cv.INTER_NEAREST)

    return mask


def global_contrast_mask(img):
    """Create a 'fat pixel' B&W mask, representing areas of high contrast as white, and low contrast as black"""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bw_img = cv.blur(gray, (5, 5))
    bw_img = cv.adaptiveThreshold(
        bw_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 2
    )
    _, bw_img = cv.threshold(bw_img, 120, 255, cv.THRESH_BINARY_INV)

    # Use the "hit or miss" algorithm to get rid of noise:
    kernel = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype="int")
    strays = cv.morphologyEx(bw_img, cv.MORPH_HITMISS, kernel)
    output = cv.bitwise_xor(strays, bw_img)

    # Scale down and back up
    scale = 8
    orig_size = img.shape
    new_width, new_height = int(img.shape[1] / scale), int(img.shape[0] / scale)
    small = cv.resize(output, (new_width, new_height), interpolation=cv.INTER_AREA)
    _, small = cv.threshold(small, 250, 255, cv.THRESH_BINARY_INV)

    final = cv.resize(
        small, (orig_size[1], orig_size[0]), interpolation=cv.INTER_NEAREST
    )

    return final


def invert_mask(img):
    """
    Calculate areas of light and shadow in order to provide an inversion mask to use on the ASCII version
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # TODO: remove this hardcoding
    small_width = 57
    small_height = 45

    # Make it small
    img = cv.resize(img, (small_width, small_height), interpolation=cv.INTER_LINEAR)

    res, img = cv.threshold(img, 25, 255, cv.THRESH_BINARY)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    # Scale it back up
    scale = 8
    img = cv.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv.INTER_NEAREST,
    )

    return img


def get_as_csv(img):
    """Return the pixel image of a binary image as a single string of 0's and 1's"""
    csv = []

    for pixel in img.flat:
        pixel = pixel if pixel == 0 else 1
        csv.append(str(pixel))

    csv_str = "\t".join(csv)

    return csv_str


def get_images_from_csv(csv:str, size=(8,8)):
    """Reverse of the previous function, it returns a grayscale image per row from a string of CSV data,
    formatted as above"""

    images = []

    for row in csv.split('\n'):
        img = np.full(size, 0, dtype=np.uint8)

        # discard first entry, since it's the ASCII label
        row_parts = row.split('\t', 1)
        row = row_parts[1]

        for i, pixel in enumerate(row.split('\t')):
            x = i % size[0]
            y = math.floor(i / size[1])
            img[y, x] = int(pixel) * 255

        images.append(img)

    return images
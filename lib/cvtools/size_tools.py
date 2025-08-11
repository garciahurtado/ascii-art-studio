from dataclasses import dataclass

import numpy as np
import cv2 as cv
from numpy.lib.stride_tricks import as_strided as as_strided
import math

@dataclass
class Dimensions:
    width: int
    height: int

def resize_with_padding(img, size, padColor=0, block_size=8):
    """
    Resize image with padding to maintain aspect ratio and ensure dimensions are multiples of block_size.
    
    Args:
        img: Input image
        size: Target size as (width, height)
        padColor: Color to use for padding
        block_size: Size of character blocks (default 8x8)
        
    Returns:
        Resized and padded image with dimensions that are multiples of block_size
    """
    height, width = img.shape[:2]
    frame_width, frame_height = size
    
    # Ensure target dimensions are multiples of block_size
    frame_width = (frame_width // block_size) * block_size
    frame_height = (frame_height // block_size) * block_size
    
    # interpolation method
    if height > frame_height or width > frame_width:  # shrinking image
        interp = cv.INTER_AREA
    else:  # stretching image
        interp = cv.INTER_CUBIC

    # aspect ratio of image
    img_aspect = width / height
    frame_aspect = frame_width / frame_height

    # compute scaling and pad sizing
    if img_aspect > frame_aspect:
        # Horizontal bars
        new_width = frame_width
        new_height = max(block_size, (frame_width // img_aspect) // block_size * block_size)
        pad_horizontal = (frame_height - new_height) // 2
        pad_vertical = 0
        
        # Ensure padding is even and dimensions are correct
        pad_horizontal = (pad_horizontal // block_size) * block_size
        new_height = frame_height - 2 * pad_horizontal
        
    elif img_aspect < frame_aspect:
        # Vertical bars
        new_height = frame_height
        new_width = max(block_size, (frame_height * img_aspect) // block_size * block_size)
        pad_vertical = (frame_width - new_width) // 2
        pad_horizontal = 0
        
        # Ensure padding is even and dimensions are correct
        pad_vertical = (pad_vertical // block_size) * block_size
        new_width = frame_width - 2 * pad_vertical
    else:  # square image
        new_height, new_width = frame_height, frame_width
        pad_horizontal, pad_vertical = 0, 0

    # set padding color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv.resize(img, (int(new_width), int(new_height)), interpolation=interp)
    scaled_img = cv.copyMakeBorder(
        scaled_img,
        int(pad_horizontal), int(pad_horizontal),
        int(pad_vertical), int(pad_vertical),
        borderType=cv.BORDER_CONSTANT,
        value=padColor
    )
    
    # Final check to ensure dimensions are correct
    h, w = scaled_img.shape[:2]
    if h % block_size != 0 or w % block_size != 0:
        # If we still have incorrect dimensions, do a final crop
        h = (h // block_size) * block_size
        w = (w // block_size) * block_size
        scaled_img = scaled_img[:h, :w]

    return scaled_img


def resize_grayscale(img, size):
    """ Resize a grayscale image """
    output = cv.resize(img, size, interpolation=cv.INTER_AREA)
    return output

def adjust_img_size(in_dims: Dimensions, min_dims: Dimensions, max_dims: Dimensions):
    """
    Take the *smallest* dimension of the input image and match it to the equivalent dimension of the output image. Calculate
    the aspect ratio of the input image, and use that ratio to calculate the other dimension.
    This should minimize letterboxing during resizing, and work for both portrait and landscape images, without having to manually
    specify different output dimensions for each aspect ratio.
    """ 
    img_aspect_ratio = in_dims.width / in_dims.height

    if in_dims.width < in_dims.height:    # Portrait
        new_width = min_dims.width
        new_height = int(new_width / img_aspect_ratio)
        if new_height < min_dims.height:
            new_height = min_dims.height
            new_width = int(new_height * img_aspect_ratio)
        elif new_height > max_dims.height:
            new_height = max_dims.height
            new_width = int(new_height * img_aspect_ratio)

    else:                               # Landscape or Square
        new_height = min_dims.height
        new_width = int(new_height * img_aspect_ratio)
        if new_width < min_dims.width:
            new_width = min_dims.width
            new_height = int(new_width / img_aspect_ratio)
        elif new_width > max_dims.width:
            new_width = max_dims.width
            new_height = int(new_width / img_aspect_ratio)

    # Adjust to nearest multiple of 8
    new_height = math.ceil(new_height / 8) * 8
    new_width = math.ceil(new_width / 8) * 8

    return new_width, new_height


def as_blocks(image, size=(8,8), flatten=False):
    return sliding_window(image, size, flatten=flatten)

""" Numpy manipulation functions"""

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(input_array, window_size, slide_size=None, flatten=False):
        '''
        Return a sliding window over input_array in any number of dimensions

        Parameters:
            input_array  - an n-dimensional numpy array

            window_size - an int (a is 1D) or tuple (a is 2D or greater) representing the size
                 of each dimension of the window

            slide_size - an int (a is 1D) or tuple (a is 2D or greater) representing the
                 amount to slide the window in each dimension. If not specified, it
                 defaults to ws.

            flatten - if True, all slices are flattened, otherwise, there is an
                      extra dimension for each dimension of the input.

        Returns
            an array containing each n-dimensional window from input_array

        from http://www.johnvinyard.com/blog/?p=268
        '''

        if None is slide_size:
            # ss was not provided. the windows will not overlap in any direction.
            slide_size = window_size

        window_size = norm_shape(window_size)
        slide_size = norm_shape(slide_size)

        # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
        # dimension at once.
        window_size = np.array(window_size)
        slide_size = np.array(slide_size)
        shape = np.array(input_array.shape)

        # ensure that ws, ss, and a.shape all have the same number of dimensions
        ls = [len(shape), len(window_size), len(slide_size)]
        if 1 != len(set(ls)):
            raise ValueError( \
                'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

        # ensure that ws is smaller than a in every dimension
        if np.any(window_size > shape):
            raise ValueError(
                'ws cannot be larger than a in any dimension. a.shape was %s and ws was %s' % (
                str(input_array.shape), str(window_size)))

        # how many slices will there be in each dimension?
        newshape = norm_shape(((shape - window_size) // slide_size) + 1)

        # the shape of the strided array will be the number of slices in each dimension
        # plus the shape of the window (tuple addition)
        newshape += norm_shape(window_size)

        # the strides tuple will be the array's strides multiplied by step size, plus
        # the array's strides (tuple addition)
        newstrides = norm_shape(np.array(input_array.strides) * slide_size) + input_array.strides
        strided = as_strided(input_array, shape=newshape, strides=newstrides)

        if not flatten:
            return strided

        # Collapse strided so that it has one more dimension than the window.  I.e.,
        # the new array is a flat list of slices.
        meat = len(window_size) if window_size.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if window_size.shape else ()
        dim = firstdim + (newshape[-meat:])

        # remove any dimensions with size 1
        dim = list(filter(lambda i: i != 1, dim))

        return strided.reshape(dim)



def is_empty(img):
    white_pixels = np.count_nonzero(img)

    if white_pixels == 0:
        # No white pixels, they must be all black
        return True
    else:
        return False


def is_full(img):
    white_pixels = np.count_nonzero(img)
    total_pixels = img.shape[0] * img.shape[1]

    if white_pixels == total_pixels:
        return True
    else:
        return False

def is_almost_empty(img, threshold):
    """ Returns true as long as the image has no more than 'threshold' black pixels"""

    white_pixels = np.count_nonzero(img)

    if white_pixels <= threshold:
        return True
    else:
        return False

def is_almost_full(img, threshold):
    """ Returns true as long as the image has no more than 'threshold' white pixels"""
    white_pixels = np.count_nonzero(img)
    total_pixels = img.shape[0] * img.shape[1]

    if white_pixels >= (total_pixels - threshold):
        return True
    else:
        return False
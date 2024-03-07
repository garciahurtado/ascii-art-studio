from functools import reduce

import imagehash as imhash
import cv2 as cv
import scipy.spatial.distance as distance
import numpy as np

def image_dhash(image):
    """ Input must be a PIL image"""
    hash = imhash.dhash(image, hash_size=64)
    return hash

def image_phash(image):
    """ Input must be a PIL image"""
    hash = imhash.phash(image, hash_size=64)  #
    return hash

def image_color_hash(image):
    """ Input must be a PIL image"""
    hash = imhash.colorhash(image, 3)
    return hash


def image_cv_hash(image):
    hash_maker = cv.img_hash.PHash_create()
    hash = hash_maker.compute(image)
    # hash_xor = reduce(lambda x, y: x ^ y, hash[0])
    return hash

def get_hamming_distance(hash1, hash2):
    dist_ham = cv.norm(hash1 - hash2, normType=cv.NORM_HAMMING)
    return dist_ham

def get_distance(img1, img2):
    dist = distance.cdist(img1, img2, 'cityblock')
    total = np.sum(dist)
    return total

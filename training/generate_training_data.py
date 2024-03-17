"""Creates training images for training of a neural network that will learn to match b&w image blocks
to ASCII characters"""
import csv
import glob
import math
import os
import random

import arrow
from collections import Counter

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from charset import Charset
from ascii import FeatureAsciiConverter
from cvtools.processing_pipeline import ProcessingPipeline

IN_DIR = 'images/in/'
OUT_DIR = 'images/out/'
#WIDTH, HEIGHT = 496, 360  # For images
WIDTH, HEIGHT = 584, 360  # For video 8x8 (and video stills)
# WIDTH, HEIGHT = 1168, 736  # For video 8x16
all_used_chars = [] # For stats

charset_name = 'c64.png'
char_width, char_height = 8, 8
charset = Charset(char_width, char_height)
charset.load(charset_name)


def create_single_image(filename, index, csv=True, labels=False, skip_blank_rows=True, random_inverted=False, double_inverted=True):

    if os.path.isfile(filename):
        print(f'Processing {filename}...')

        in_img = cv.imread(filename)

        out_filename = f"{index:06d}"
        convert_image(in_img, labels, out_filename, random_inverted=random_inverted)

        out_filename = f"{index:06d}-inv"
        convert_image(in_img, labels, out_filename, random_inverted=random_inverted, is_inverted=True)


def convert_image(in_img, labels, filename, random_inverted=False, is_inverted=False):
    out_file_contrast = OUT_DIR + f'{filename}-contrast.png'
    out_file_ascii = OUT_DIR + f'{filename}-ascii.png'

    converter = FeatureAsciiConverter(charset)
    converter.char_width = char_width
    converter.char_height = char_height
    pipeline = ProcessingPipeline()
    pipeline.converter = converter
    pipeline.img_width = WIDTH
    pipeline.img_height = HEIGHT

    # pipeline.contrast_img = cv.bitwise_not(pipeline.contrast_img)

    # Half of the images will be randomly inverted. This will ensure that the characters in the inverted
    # half of the charset are also represented.
    # if random_inverted and random.choice([True, False]):
    #     pipeline.run(in_img)
    # else:
    #     pipeline.run(in_img, invert=True)

    if not is_inverted:
        pipeline.run(in_img)
    else:
        pipeline.run(in_img, invert=True)

    # Save the high contrast image
    cv.imwrite(out_file_contrast, pipeline.contrast_img)
    print(f'{out_file_contrast} created')

    # Save the ASCII converted image, as text
    cv.imwrite(out_file_ascii, pipeline.ascii)
    print(f'{out_file_ascii} created')

    # Collect stats for histogram of per character uses
    all_used_chars.extend(converter.used_chars)

    if csv:
        csv_data = converter.get_csv_data()
        out_file_data = OUT_DIR + f'{filename}-ascii-data.txt'
        data_file = open(out_file_data, "w")
        data_file.write(csv_data)
        data_file.close()
        print(f'{out_file_data} created')

        out_file_table = OUT_DIR + f'{filename}-ascii-table.png'
        index_table = make_charset_index_table(charset)
        cv.imwrite(out_file_table, index_table)
        print(f'{out_file_table} created')

        return csv_data

    if labels:
        return converter.get_label_data()

def create_training_images(csv=True, start_index=0):
    """
    :Bool csv: Whether to create CSV files of the ASCII characters used in each image
    :int start_index: Pass something other than zero to avoid filenaming conflicts
    :rtype: None
    """
    entries = os.listdir(IN_DIR)
    num_threads = 8
    start_time = arrow.utcnow()

    with Pool(num_threads) as p:
        all_params = []

        for index, file in enumerate(entries):
            full_path = os.path.join(IN_DIR, file)

            if os.path.isdir(full_path):
                continue # Skip directories

            params = [full_path, start_index + index, csv]
            all_params.append(params)

        p.starmap(create_single_image, all_params)

    end_time = arrow.utcnow()
    time_diff = end_time - start_time
    num_entries = len(all_params)
    print(f"{num_entries} images processed by {num_threads} threads in: {time_diff}")

    # Display character histogram
    show_hist = True
    if(show_hist):
        counts = Counter(all_used_chars)

        plt.rc('axes', titlesize=8)

        cols, rows = 18, 18
        fig, axs = plt.subplots(rows, cols, figsize=(16,8))
        plt.subplots_adjust(left=None, bottom=-1)

        row_id = col_id = 0
        bar_color = (0, 1, 0, 1)

        for (char, count) in counts.items():
            char_img = cv.cvtColor(char.img, cv.COLOR_GRAY2RGB)

            axs[row_id, col_id].bar([0], count, color=bar_color)
            axs[row_id, col_id].set_title(f'#{char.index}')
            axs[row_id, col_id].imshow(char_img, cmap='gray', interpolation='nearest')
            axs[row_id, col_id].get_xaxis().set_visible(False)
            axs[row_id, col_id].get_yaxis().set_visible(False)

            col_id += 1

            if(col_id >= cols):
                col_id = 0
                row_id += 1

        plt.show()


def make_charset_index_table(charset, char_width=8, char_height=8):
    """Generate an image showing each character and their corresponding index in the charset they were loaded from.
    This is needed for model training, as those indexes will become the labels"""

    size = (len(charset) * char_height, char_width * 5)
    img = np.zeros(size, dtype=np.uint8)

    for i, character in enumerate(charset):
        img[i * char_height: (i+1) * char_height, 0:char_width] = character.img
        color = (255,255,255)
        cv.putText(img, str(character.index), (char_width + 3, ((i+1) * char_height) - 1), cv.FONT_HERSHEY_PLAIN, .75, color)

    return img

def make_charset_sprite(charset):
    '''Generate a single sprite containing the images of every character in the charset, to be used as image labels for
    Tensorboard'''

    char_width, char_height = 8, 16
    cols = math.ceil(math.sqrt(len(charset)))
    size = (cols * char_height, cols * char_width)
    img = np.zeros(size, dtype=np.uint8)

    for i, character in enumerate(charset):
        col = i % cols
        row = math.floor(i / cols)

        img[row * char_height : (row+1) * char_height, col * char_width : (col + 1) * char_width] = character.img

    return img

def make_histogram():
    '''Given a list of datafiles, create a histogram of the label frequency, to identify labels with low representation'''

    entries = glob.glob(OUT_DIR + "*data.txt")

    num_threads = 16
    start_time = arrow.utcnow()

    num_entries = 0
    unsorted = []
    freq_map = {}
    max_val = 0

    with Pool(num_threads) as p:

        for index, file in enumerate(entries):

            if os.path.isdir(file):
                continue  # Skip directories

            # Read file contents
            with open(file, newline='') as csvfile:

                for row in csv.reader(csvfile, delimiter='\t', quotechar='|'):
                    label = row[0]
                    unsorted.append(label)

                    if label in freq_map.keys():
                        freq_map[label] += 1

                        if freq_map[label] > max_val:
                            max_val = freq_map[label]
                    else:
                        freq_map[label] = 1

            num_entries += 1
            print(file + ' processed')

    # invert values in freq map, to produce weights
    freq_map = {k: (1 / v) * max_val for k, v in freq_map.items()}

    print(freq_map)

    end_time = arrow.utcnow()
    time_diff = end_time - start_time

    print(f"{num_entries} data files processed by {num_threads} threads in: {time_diff}")

    # ----


    plt.hist(unsorted, bins=len(unsorted))
    plt.xticks(range(1))
    plt.title('Label Frequency')
    plt.show()



def eval_determinism():
    """The algorithm which creates the ASCII labels used in training must produce consistent results in order
    for the training set to be 100% accuracy trainable. This function tests the algorithm for consistent results in
    labeling."""

    files = os.listdir(IN_DIR)
    start_index = 0
    num_threads = 4
    rounds_per_image = 2
    start_time = arrow.utcnow()
    uncertain_labels = []
    all_labels = []

    with Pool(num_threads) as p:
        last_labels = None
        all_params = []

        for index, file in enumerate(files):
            full_path = os.path.join(IN_DIR, file)

            if os.path.isdir(full_path):
                continue  # Skip directories

            for round in range(1, rounds_per_image + 1):
                params = [full_path, charset, start_index + index, False, True] # Return Labels, not CSV
                all_params.append(params)

                labels = p.starmap(create_single_image, all_params)
                labels = labels[0]

                # Keep track of any labels that differ between this conversion and the last
                if last_labels and len(last_labels) == len(labels):
                    for idx, label in enumerate(labels):
                        if labels[idx] != last_labels[idx]:
                            uncertain_labels.append(labels[idx])
                            uncertain_labels.append(last_labels[idx])


                last_labels = labels
                all_labels.append(labels)

    print("Uncertain labels: " + uncertain_labels)

if __name__ == "__main__":
    create_training_images()
    # make_histogram()


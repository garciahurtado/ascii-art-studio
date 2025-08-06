"""Creates training images for training of a neural network that will learn to match b&w image blocks
to ASCII characters"""
import csv
import glob
import math
import os
import argparse
from dataclasses import dataclass

import arrow
from collections import Counter
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool

from charset import Charset
from ascii import FeatureAsciiConverter
from cvtools.processing_pipeline import ProcessingPipeline
from cvtools.size_tools import Dimensions, adjust_img_size

IN_DIR = 'images/in/'
IN_DIR_DATA = 'tmp_data/'
OUT_DIR = 'images/out/'

# You have to generate the training data in two passes, first for images, then for video
# WIDTH, HEIGHT = 496, 360  # For images
WIDTH, HEIGHT = 384, 424  # For images (joker)

# Ideally the dimensions would be automatically determined from the input image like so:


# half size will produce 25% the number of training examples as before
# WIDTH, HEIGHT = 512, 248  # For video 8x8 (and video stills)

# For 8x16 charsets only:
# WIDTH, HEIGHT = 1168, 480  # For video 8x16
# WIDTH, HEIGHT = 584, 368  # half size 8x16

all_used_chars = [] # For stats

charset_name = 'c64.png'
charset = Charset()
charset.load(charset_name)
char_width, char_height = charset.char_width, charset.char_height

# charset.write('unscii_8x8-packed.png') # pack the character set (only needed once)

def create_single_image(filename, index, min_dims: Dimensions, max_dims: Dimensions, export_csv=True, color_only=False, labels=False, double_inverted=True):

    if os.path.isfile(filename):
        print(f'Processing {filename}...')

        in_img = cv.imread(filename)

        out_filename = f"{index:06d}"
        convert_image(in_img, labels, out_filename, min_dims, max_dims, export_csv=export_csv, color_only=color_only)

        if double_inverted:
            out_filename = f"{index:06d}-inv"
            convert_image(in_img, labels, out_filename, min_dims, max_dims, export_csv=export_csv, color_only=color_only, is_inverted=True)

        return True
    else:
        return False


def convert_image(in_img, labels, filename, min_dims, max_dims, export_csv=True, color_only=False, is_inverted=False):
    out_file_color = OUT_DIR + f'{filename}-color.png'
    out_file_contrast = OUT_DIR + f'{filename}-contrast.png'
    out_file_ascii = OUT_DIR + f'{filename}-ascii.png'

    in_img_width, in_img_height = in_img.shape[1], in_img.shape[0]
    in_dims = Dimensions(in_img_width, in_img_height)

    converter = FeatureAsciiConverter(charset)
    pipeline = ProcessingPipeline(brightness=100, contrast=3.0)
    pipeline.converter = converter
    pipeline.img_width, pipeline.img_height = adjust_img_size(in_dims, min_dims, max_dims)

    final_img = pipeline.run(in_img, invert=is_inverted)

    # Save the final color image
    cv.imwrite(out_file_color, final_img)
    print(f'> {out_file_color} created')

    if color_only:
        return

    # Save the high contrast image
    cv.imwrite(out_file_contrast, pipeline.contrast_img)
    print(f'> {out_file_contrast} created')

    # Save the ASCII converted image, as text
    cv.imwrite(out_file_ascii, pipeline.ascii)
    print(f'> {out_file_ascii} created')

    # Collect stats for histogram of per character uses
    all_used_chars.extend(converter.used_chars)

    if export_csv:
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

def create_training_data(min_dims: Dimensions, max_dims: Dimensions, export_csv=True, color_only=False, start_index=0, double_inverted=True):
    """
    :Bool export_csv: Whether to create CSV files of the ASCII characters used in each image
    :int start_index: Pass something other than zero to avoid filenaming conflicts
    :rtype: None
    """
    all_files = os.listdir(IN_DIR)

    # only process .png all_files
    all_files = [entry for entry in all_files if entry.endswith('.png')]
    num_img = len(all_files)

    if num_img < 1:
        raise ValueError(f'No .PNG images found in {IN_DIR}')

    # Don't use more threads than the number of images
    num_threads = 16 if num_img > 16 else num_img

    start_time = arrow.utcnow()
    labels = False

    with Pool(num_threads) as p:
        all_params = []

        for index, file in enumerate(all_files):
            full_path = os.path.join(IN_DIR, file)

            if os.path.isdir(full_path):
                continue # Skip directories

            params = [full_path, start_index + index, min_dims, max_dims, export_csv, color_only, labels, double_inverted]
            all_params.append(params)

        p.starmap(create_single_image, all_params)

    end_time = arrow.utcnow()
    time_diff = end_time - start_time
    num_entries = len(all_params)
    print(f"*** DONE: {num_entries} images processed by {num_threads} threads in: {time_diff} ***")

    """ Once the dataset is complete, create the metadata file:
        name: ascii_c64
    version: 1.0.0
    created: 2025-08-05
    description: "ASCII art dataset from C64 character set"
    license: MIT
    source: "https://example.com/source"
    features:
      - name: image
        description: "8x8 grayscale character patches"
        shape: [8, 8]
        dtype: float32
      - name: label
        description: "Character class index"
        dtype: int64
    splits: [train, val, test]
    statistics:
      num_samples: 10000
      class_distribution: "path/to/distribution.json"
      """
    this_script = os.path.basename(__file__)
    dataset_name = this_script.split('.')[0]
    version = "0.0.1"

    yaml_dict = {
        "name": dataset_name,
        "version": version,
        "created": date_time,
        "description": f"ASCII art dataset generated from {charset_name} character set",
        "license": "MIT",
        "source": "https://github.com/garciahurtad/ascii_movie_pytorch",
        "features": [
            {
                "name": "image",
                "description": "8x8 grayscale character patches",
                "shape": [8, 8],
                "dtype": "float32"
            },
            {
                "name": "label",
                "description": "Character class index",
                "dtype": "int64"
            }
        ],
        "train_split": "n/a",
        "statistics": {
            "num_samples": num_entries,
            "class_distribution": "n/a"
        },
        "cmdline": f"python {this_script} {IN_DIR} {OUT_DIR}"
    }

    # Display character histogram
    # show_histogram(all_used_chars)


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

def show_histogram(all_used_chars):
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

        if col_id >= cols:
            col_id = 0
            row_id += 1

    plt.show()

def make_charset_index_table(charset):
    """Generate an image showing each character and their corresponding index in the charset they were loaded from.
    This is needed for model training, as those indexes will become the labels"""
    char_width, char_height = charset.char_width, charset.char_height
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

def _read_csv_file_for_shuffling(file_path):
    """Helper function for shuffle_and_rebatch_csvs to read a single CSV file."""
    with open(file_path, 'r', newline='') as f:
        return list(csv.reader(f, delimiter='\t', quotechar='|'))

def _write_rebatched_files(data, directory, prefix, rows_per_file):
    """Helper function to write rows to batched CSV files."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_output_files = math.ceil(len(data) / rows_per_file)
    print(f"Writing {len(data)} rows into {num_output_files} files in '{directory}'...")

    for i in range(num_output_files):
        start_index = i * rows_per_file
        end_index = start_index + rows_per_file
        chunk = data[start_index:end_index]

        out_filename = os.path.join(directory, f"{prefix}_{i:04d}.csv")
        with open(out_filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|')
            writer.writerows(chunk)


def shuffle_and_rebatch_csvs(in_dir, out_dir, rows_per_file=1776, split_ratio=0.8):
    """
    Reads all *-ascii-data.txt files from in_dir, shuffles the rows,
    and writes them to new CSV files in out_dir, each with a fixed number of rows.
    The data is split into 'train' and 'test' subdirectories based on the split_ratio.

    This process is handled in a multi-threaded fashion to accelerate file I/O.
    """
    print("--- Starting CSV shuffling and re-batching ---")
    start_time = arrow.utcnow()

    csv_files = glob.glob(os.path.join(in_dir, "*-ascii-data.txt"))
    if not csv_files:
        print(f"No '*-ascii-data.txt' files found in {in_dir}. Aborting.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    # Use a thread pool to read files in parallel
    with Pool() as p:
        results = p.map(_read_csv_file_for_shuffling, csv_files)

    all_rows = [row for file_rows in results for row in file_rows]

    print(f"Total rows read: {len(all_rows)}")

    # Shuffle the data in-place
    print("Shuffling data...")
    random.shuffle(all_rows)
    print("Shuffling complete.")

    # Split data into training and testing sets
    split_index = int(len(all_rows) * split_ratio)
    train_rows = all_rows[:split_index]
    test_rows = all_rows[split_index:]

    print(f"Splitting data: {len(train_rows)} for training, {len(test_rows)} for testing.")

    # Define output directories
    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')

    # Write re-batched files
    _write_rebatched_files(train_rows, train_dir, "train_data", rows_per_file)
    _write_rebatched_files(test_rows, test_dir, "test_data", rows_per_file)

    end_time = arrow.utcnow()
    time_diff = end_time - start_time
    print(f"--- Finished in {time_diff} ---")


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
    """
    Defaults:
    """
    out_min_width, out_min_height = 384, 384        # 48x48 characters of 8x8
    out_max_width, out_max_height = 528, 528        # 64x64 characters of 8x8

    parser = argparse.ArgumentParser(description='Generate training data for the ASCII NN Converter.')
    parser.add_argument(
        '--img-only',
        help='Only create pipeline images, do not create data files',
        action='store_true',
        default=False)
    parser.add_argument(
        '--color-only',
        help='Only generate the final color image, do not create other images or data files',
        action='store_true',
        default=False)
    parser.add_argument(
        '--start',
        help='First number to be used in naming the output files. The rest will be incremented',
        action='store',
        default=0)
    parser.add_argument(
        '--width',
        help='Width of the output image. If it is not a multiple of the char width, it will be clipped',
        action='store',
        default=out_min_width)
    parser.add_argument(
        '--height',
        help='Height of the output image. If it is not a multiple of the char height, it will be clipped',
        action='store',
        default=out_min_height)
    parser.add_argument(
        '--inverted',
        help='Generate a second set of inverted images and data files (will not generate by default)',
        action='store_true',
        default=False)
    parser.add_argument(
        '--shuffle',
        help='Shuffle and re-batch existing *-ascii-data.txt files',
        action='store_true',
        default=False)
    parser.add_argument(
        '--split-ratio',
        help='Split ratio for training and testing sets',
        action='store',
        default=0.8)

    args = parser.parse_args()
    img_only = args.img_only
    color_only = args.color_only
    start_index = args.start
    min_dims = Dimensions(width=int(args.width), height=int(args.height))
    max_dims = Dimensions(width=out_max_width, height=out_max_height)
    double_inverted = args.inverted
    split_ratio = float(args.split_ratio)

    if args.shuffle:
        shuffle_and_rebatch_csvs(IN_DIR_DATA, os.path.join(IN_DIR_DATA, 'shuffled'), split_ratio=split_ratio)
    else:
        create_training_data(min_dims, max_dims, export_csv=(not img_only), color_only=color_only, start_index=int(start_index), double_inverted=double_inverted)

"""Creates training images for training of a neural network that will learn to match b&w image blocks
to ASCII characters"""
import csv
import glob
import math
import os
import argparse
import sys
import arrow
from collections import Counter
import random

from datasets.data_augment import create_augmented_images
import datasets.data_utils as data_utils
import cv2 as cv
import matplotlib.pyplot as plt

from multiprocessing import Pool

""" Project imports """

from const import PROJECT_ROOT, DATASETS_ROOT, INK_BLUE
from charset import Charset

from ascii import FeatureAsciiConverter
from const import INK_GREEN, INK_YELLOW
from const import out_max_height, out_max_width, out_min_height, out_min_width
from cvtools.processing_pipeline import ProcessingPipeline
from cvtools.size_tools import Dimensions, adjust_img_size
from datasets.ascii_dataset import AsciiDataset
from datasets.data_utils import starmap_with_kwargs
from debugger import printc

DATA_DIR = 'processed'
IN_IMG_DIR = os.path.join('images', 'in')
OUT_IMG_DIR = os.path.join('images', 'out')
IN_DATA_DIR = 'dataset_tmp'

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

def process_image(filename, data_path, index, min_dims: Dimensions, max_dims: Dimensions, export_csv=True,
                  color_only=False,
                  double_inverted=True, return_labels=False):

    if os.path.isfile(filename):
        print(f'Processing {filename}...')
        in_img = cv.imread(filename)

        out_filename = f"{index:06d}"
        num_samples = convert_image(in_img, out_filename, data_path, min_dims, max_dims, export_csv=export_csv,
                                    color_only=color_only,
                      return_labels=return_labels)

        if double_inverted:
            out_filename = f"{index:06d}-inv"
            num_samples += convert_image(in_img, out_filename, data_path, min_dims, max_dims, export_csv=export_csv,
                          color_only=color_only, is_inverted=True, return_labels=return_labels)

        return True
    else:
        raise FileNotFoundError(f'File {filename} does not exist')


def convert_image(in_img, filename, data_path, min_dims, max_dims, export_csv=True, color_only=False, is_inverted=False,
                  return_labels=False):
    # debug constants
    print(f'OUT_IMG_DIR {OUT_IMG_DIR}...')
    print(f'OUT_DATA_PATH: {data_path}')
    out_file_color = os.path.join(OUT_IMG_DIR, f'{filename}-color.png')
    out_file_contrast = os.path.join(OUT_IMG_DIR, f'{filename}-contrast.png')
    out_file_ascii = os.path.join(OUT_IMG_DIR, f'{filename}-ascii.png')
    out_file_data = os.path.join(data_path, f'{filename}-ascii-data.csv')
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
        data_file = open(out_file_data, "w")
        data_file.write(csv_data)
        data_file.close()
        print(f'{out_file_data} created')

        return csv_data

    if return_labels:
        return converter.get_label_data()
    else:
        num_samples = len(all_used_chars)
        return num_samples


def create_training_data(min_dims: Dimensions, max_dims: Dimensions, export_csv=True, color_only=False,
                         start_index=0, double_inverted=True, dataset_name=None):
    """
    Create training data for a dataset of images. Each image is processed in parallel using a configurable number of
    threads. Optionally, the images can be inverted before processing and the intermin images can be saved along with
    the dataset.
    """
    all_files = os.listdir(IN_IMG_DIR)
    total_num_samples = 0

    # only process .png all_files
    all_files = [entry for entry in all_files if entry.endswith('.png')]
    num_imgs = len(all_files)

    if num_imgs < 1:
        raise ValueError(f'No .PNG images found in {IN_IMG_DIR}!!')

    # Don't use more threads than the number of images
    num_threads = 16 if num_imgs > 16 else num_imgs

    start_time = arrow.utcnow()

    DATASET_NAME_PATH = os.path.join(DATASETS_ROOT, dataset_name)
    out_data_dir = os.path.realpath(os.path.join(DATASET_NAME_PATH, 'processed/'))

    print(f'Creating training data for {num_imgs} images in {OUT_IMG_DIR}...')

    """ Make sure that the output directories exist and are empty before processing new data """
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    else:
        for file in os.listdir(out_data_dir):
            full_path = os.path.join(out_data_dir, file)
            if os.path.isfile(full_path):
                raise FileExistsError(
                    f'{out_data_dir} is not empty! Please remove any files in this directory before processing new data.')

    if not os.path.exists(OUT_IMG_DIR):
        os.makedirs(OUT_IMG_DIR)
    else:
        for file in os.listdir(OUT_IMG_DIR):
            full_path = os.path.join(OUT_IMG_DIR, file)
            if os.path.isfile(full_path) and file.endswith('.png'):
                raise FileExistsError(
                    f'Directory {OUT_IMG_DIR} is not empty! Please remove any .PNG files in this directory before processing new data.')

    # Create an image of the charset that we will be using for this dataset generation, so we can save it for reference
    # along with the training data
    out_charset_table = OUT_IMG_DIR + f'{dataset_name}-ascii-table.png'
    index_table = Charset.make_charset_index_table(charset)
    cv.imwrite(out_charset_table, index_table)
    print(f'ASCII index table: {out_charset_table} created')

    with Pool(num_threads) as p:
        all_args = []
        all_kwargs = []

        for index, file in enumerate(all_files):
            in_img_path = os.path.join(IN_IMG_DIR, file)

            if os.path.isdir(in_img_path):
                continue  # Skip directories

            _args = [
                in_img_path,
                out_data_dir,
                start_index + index,
                min_dims,
                max_dims
            ]
            all_args.append(_args)

            _kwargs = {
                'export_csv': export_csv,
                'color_only': color_only,
                'double_inverted': double_inverted
            }
            all_kwargs.append(_kwargs)

        return_values = starmap_with_kwargs(p, process_image, all_args, all_kwargs)

        """ process_image returns an integer with the number of samples generated. starmap will return a list of those
        results """
        for return_value in return_values:
            if return_value:
                total_num_samples += return_value

    end_time = arrow.utcnow()
    time_diff = end_time - start_time
    num_files_processed = len(all_args)  # weird way to get the number of files processed
    print(f"*** DONE: {num_files_processed} images processed by {num_threads} threads in: {time_diff} ***")

    """ Now that the dataset is complete, create the metadata file """
    dataset = AsciiDataset(dataset_name)

    num_samples = total_num_samples
    new_version = save_metadata(dataset, num_files_processed, num_samples)

    printc(f"Saved dataset with version {new_version} to {out_data_dir}", INK_BLUE)


def save_metadata(dataset: AsciiDataset, num_files_processed, num_samples):
    if not os.path.isfile(dataset.metadata_path):
        # Create the metadata file if it doesn't exist
        new_file = open(dataset.metadata_path, 'w')
        new_file.close()
        printc(f"METADATA FILE CREATED FOR NEW DATASET '{dataset_name}' at {dataset.metadata_path}", INK_GREEN)

    dataset.load_metadata()

    this_script = os.path.basename(__file__)
    date_time = arrow.utcnow().format('YYYY-MM-DD_HH:mm:ss')

    # if version is already set, increment it
    if 'version' in dataset.metadata.keys():
        # split version into major, minor, and patch
        version_major, version_minor, version_patch = dataset.metadata['version'].split('.')
        version_patch = int(version_patch) + 1
        version = str(f"{version_major}.{version_minor}.{version_patch}")
    else:
        version = "0.0.1"

    sys_args = f"{sys.argv[1:]}"

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
            "num_source_files": num_files_processed,
            "num_samples": num_samples,
            "class_distribution": "n/a"
        },
        "cmdline": f"python {this_script} {sys_args}"
    }

    # merge dictionaries
    dataset.metadata.update(yaml_dict)
    dataset.save_metadata()
    return version

def make_histogram():
    '''Given a list of datafiles, create a histogram of the label frequency, to identify labels with low representation'''

    entries = glob.glob(OUT_IMG_DIR + "*data.txt")

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

def _read_csv_file_for_shuffling(file_path):
    """Helper function for shuffle_and_rebatch_csvs to read a single CSV file."""
    with open(file_path, 'r', newline='') as f:
        return list(csv.reader(f, delimiter='\t', quotechar='|'))


def _write_rebatched_files(data, directory, rows_per_file):
    """Helper function to write rows to batched CSV files."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_output_files = math.ceil(len(data) / rows_per_file)
    print(f"Writing {len(data)} rows into {num_output_files} files in '{directory}'...")

    for i in range(num_output_files):
        start_index = i * rows_per_file
        end_index = start_index + rows_per_file
        chunk = data[start_index:end_index]

        out_filename = os.path.join(directory, f"{i:06d}-ascii-data.csv")
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

    csv_files = glob.glob(os.path.join(in_dir, "*-ascii-data.csv"))
    if not csv_files:
        print(f"No '*-ascii-data.csv' files found in {in_dir}. Aborting.")
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
    # check that the directories exist and are empty
    if os.path.exists(train_dir):
        if len(os.listdir(train_dir)) > 0:
            raise FileExistsError(f"Directory {train_dir} is not empty. Please delete the contents before re-running.")
    if os.path.exists(test_dir):
        if len(os.listdir(test_dir)) > 0:
            raise FileExistsError(f"Directory {test_dir} is not empty. Please delete the contents before re-running.")

    # Write re-batched files
    _write_rebatched_files(train_rows, train_dir, rows_per_file)
    _write_rebatched_files(test_rows, test_dir, rows_per_file)

    end_time = arrow.utcnow()
    time_diff = end_time - start_time
    print(f"--- Finished in {time_diff} ---")


def eval_determinism():
    """The algorithm which creates the ASCII labels used in training must produce consistent results in order
    for the training set to be 100% accuracy trainable. This function tests the algorithm for consistent results in
    labeling."""

    files = os.listdir(IN_IMG_DIR)
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
            full_path = os.path.join(IN_IMG_DIR, file)

            if os.path.isdir(full_path):
                continue  # Skip directories

            for round in range(1, rounds_per_image + 1):
                params = [full_path, charset, start_index + index, False, True] # Return Labels, not CSV
                all_params.append(params)

                labels = p.starmap(process_image, all_params)
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
        '--dataset',
        help='Name of the dataset to generate',
        action='store',
        default='ascii_c64')
    parser.add_argument(
        '--shuffle',
        help='Shuffle and re-batch existing *-ascii-data.txt files',
        action='store_true',
        default=False)
    parser.add_argument(
        '--split-ratio',
        help='Split ratio for training and testing sets (0.0 to 1.0: training split)',
        action='store',
        default=0.8)
    parser.add_argument(
        '--count-labels',
        help='Count the total number of examples for each label, in order to analyze the dataset distribution',
        action='store_true',
        default=False)
    parser.add_argument(
        '--subdir',
        help='Subdirectory of DATASETS_ROOT to use (ie: DATASETS_ROOT/train) (count labels only)',
        action='store',
        default='processed/train')
    parser.add_argument(
        '--augment',
        help='Augment the dataset with additional images, shifted horizontally and vertically from the original',
        action='store_true',
        default=False)

    args = parser.parse_args()
    img_only = args.img_only
    color_only = args.color_only
    start_index = args.start
    min_dims = Dimensions(width=int(args.width), height=int(args.height))
    max_dims = Dimensions(width=out_max_width, height=out_max_height)
    double_inverted = args.inverted
    dataset_name = args.dataset
    split_ratio = float(args.split_ratio)
    count_labels = args.count_labels
    subdir = args.subdir

    if args.count_labels:
        dataset_class = data_utils.get_dataset_class(dataset_name)
        dataset = dataset_class(dataset_name, subdir=subdir)
        dataset.load_metadata()
        num_classes = dataset.metadata['num_classes']
        data_utils.write_dataset_class_counts(num_classes, dataset)

        exit(0)
    elif args.shuffle:
        in_data_dir = os.path.realpath(os.path.join(DATASETS_ROOT, dataset_name, DATA_DIR))
        out_data_dir = in_data_dir  # Use the same directory as in_data_dir
        shuffle_and_rebatch_csvs(in_data_dir, out_data_dir, split_ratio=split_ratio)
        exit(0)
    elif args.augment:
        in_dir = IN_IMG_DIR
        out_dir = OUT_IMG_DIR
        if not os.path.exists(in_dir):
            printc(f"Error: {in_dir} does not exist. Please create it or choose a different directory.")

        # check that out_data_dir exists and is empty
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not data_utils.is_dir_empty(out_dir):
            printc(
                f"Error: {out_dir} already exists and is not empty. Please delete it or choose a different directory.")
            exit(1)

        num_augmented = create_augmented_images(in_dir, out_dir)

        if num_augmented:
            print(f"Created {num_augmented} augmented images in {out_dir}")
        else:
            printc(f"No source images found in {in_dir}")

        exit(0)
    else:
        create_training_data(min_dims, max_dims, export_csv=(not img_only), color_only=color_only,
                             start_index=int(start_index), double_inverted=double_inverted, dataset_name=dataset_name)
        exit(0)

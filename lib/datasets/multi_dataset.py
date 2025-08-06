import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiDataset(Dataset):
    dataset_name = None

    """ A dataset which can extract data from multiple CSV files without loading them all at once.
    It keeps track of the byte offset where each sample starts within each CSV file, in order to build an
    index which can be used for quick file seek and retrieval. """
    """ @ref https://stackoverflow.com/questions/67941749/what-is-the-fastest-way-to-load-data-from-multiple-csv-files """

    def __init__(self, train=True, batch_size=1, transform=None, target_transform=None, discard_empty_full=True,
                 **kwargs):
        super(MultiDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.char_height = 8
        self.char_width = 8

        if train:
            self.data_root = os.path.join(self.data_root, 'train')
        else:
            self.data_root = os.path.join(self.data_root, 'test')

        print("Dataset root is " + self.data_root)
        files = os.listdir(self.data_root)

        if len(files) == 0:
            raise ZeroDivisionError(f"No data files found under {self.data_root}")

        print(f"Found {len(files)} data files.")

        self.all_files = []
        self.batch_size = batch_size
        self.sample_count = 0
        self.file_indices = []  # each element is a tuple: [filename, offset]

        print("Building file indexes...")
        total_count = 0
        zero_count = 0

        for filename in files:
            self.found_first_nonzero = False
            full_path = os.path.join(self.data_root, filename)

            with open(full_path, "r", encoding='ascii') as current_file:
                offset = 0
                sample_count = 0
                for line in current_file:
                    if discard_empty_full and (not self.found_first_nonzero) and (not self.is_nonzero_sample(line)):
                        zero_count = zero_count + 1

                    if (not discard_empty_full or self.found_first_nonzero) or self.is_nonzero_sample(line):
                        self.found_first_nonzero = True
                        sample_count += 1
                        self.file_indices.append([filename, offset])

                    length = len(line.encode('ascii'))
                    offset += length + 1  # SUPERHACK: adding 1 byte to the offset due to Windows CR-LF

                self.all_files.append([full_path, sample_count])

            total_count += sample_count

        self.sample_count = total_count

        print(f"{self.sample_count} total training samples found. Ignoring {zero_count} all 0s/ all 1s samples")

    def is_nonzero_sample(self, line):
        """ We want to discard all empty samples at the beginning of a file, but only those. So we keep track of the
        first time we find a line of non-zeros, and after that we will accept anything"""
        all_ints = np.array([int(v) for v in line.split("\t")])

        # Count all columns in the row except the first (the label). Either all zeros or all ones will be discarded
        if (np.sum(all_ints[1:]) == 0) or (np.sum(all_ints[1:]) == len(all_ints[1:])):
            return False
        else:
            return True

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        full_path, offset = self.get_file_and_offset_for_index(idx)
        full_path = os.path.join(self.data_root, full_path)
        all_cols = None

        with open(full_path, "r", encoding='ascii') as csv_file:
            csv_file.seek(0)
            csv_file.seek(offset)
            line = csv_file.readline()
            all_cols = np.array([float(v) for v in line.split("\t")])
            csv_file.seek(0)

        # Extract labels (first column)
        label_data = all_cols[0]
        #label_data = label_data.reshape(-1, 1)
        label_data = torch.tensor(label_data, dtype=torch.long)

        # Extract image data
        image_data = all_cols[1:]
        image_data = image_data.reshape(1, self.char_height, self.char_width) # reorder dimensions due to Pytorch's channels 1st convention
        image_data = torch.tensor(image_data, dtype=torch.float32)

        if self.transform:
            image_data = self.transform(image_data)

        if self.target_transform:
            label_data = self.target_transform(label_data)

        sample = [image_data, label_data]

        return sample

    @lru_cache()
    def get_file_for_index(self, index):
        current_count = 0

        for file, sample_count in self.all_files:
            if current_count <= index < current_count + sample_count:
                # stop when the index we want is in the range of the sample in this file
                break  # now file_ will be the file we want
            current_count += sample_count

        # Return the filename as well as the line index relative to the start of the file
        return file, index - current_count

    @lru_cache()
    def get_file_and_offset_for_index(self, index):
        filename, offset = self.file_indices[index]
        return filename, offset

    def get_class_counts(self, test=False):
        return super().get_class_counts_from_csv(f"{self.dataset_name}_class_counts")

    def get_class_counts_from_csv(self, filename):
        ''' Loads the class counts from a CSV file previously generated '''
        filename = os.path.join(self.data_root, f"../{filename}.csv")
        class_counts = pd.read_csv(filename, header=None)
        class_counts = class_counts.to_numpy()

        return class_counts.tolist()

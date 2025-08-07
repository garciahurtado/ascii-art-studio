import math
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PreloadedDataset(Dataset):
    """A dataset which can extract data from multiple CSV files all at once."""
    """ @ref: https://biswajitsahoo1111.github.io/post/reading-multiple-csv-files-in-pytorch/ """

    def __init__(self, train=True, transform=None, target_transform=None, device=None, **kwargs):
        self.all_images = []
        self.all_labels = []
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.data_root = os.path.join(self.data_root , 'train')
        else:
            self.data_root = os.path.join(self.data_root , 'test')

        print("Dataset root is "+ self.data_root)
        files = os.listdir(self.data_root)
        num = len(files)
        if num == 0:
            raise Exception(f"No data files found under {self.data_root}")

        print(f"Found {num} data files." )
        print("Preloading samples into GPU...")

        for file in files:
            full_path = os.path.join(self.data_root , file)
            dataframe = pd.read_csv(open(full_path, 'r') , sep='\t', header=None)

            # need at least 2 byte datatype because there are 512 labels. Unfortunately,
            # Pytorch only supports int32
            data = dataframe.to_numpy(dtype=np.int16)

            # Extract labels from file (first column)
            label_data = data[:, 0:1]
            label_data = label_data.flatten()
            label_data = np.asarray(label_data, dtype='int16')
            label_data.reshape(-1, 1)

            # Extract image data from file
            image_data = data[:, 1:]
            image_data = image_data * 255  # data file stores pixels as 1 or 0, so scale up to 255
            image_data = image_data.astype('uint8')

            for label, image in zip(label_data, image_data):
                # image = image.reshape(8, 8, 1)
                image = np.asarray(image).reshape(8, 8, -1)  # Because of Pytorch's channel first convention

                if self.transform:
                    image = self.transform(image)

                label = torch.tensor(label)
                if self.target_transform:
                    label = self.target_transform(label)

                # image, label = torch.tensor(image), torch.tensor(label)
                if device:
                    image.to(device)
                    label.to(device)

                self.all_images.append(image)
                self.all_labels.append(label)

        print(f"Loaded {len(self.all_images)} samples.")

        Dataset.__init__(self)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        image = self.all_images[idx]
        label = self.all_labels[idx]

        image = np.asarray(image)


        image = image.float()
        label = label.float()

        return image, label


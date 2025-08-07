import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, train=True, batch_size=1, transform=None, target_transform=None,
                 **kwargs):
        """
        Args:
            data_dir (str): Path to the directory containing the CSV files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.data_root = os.path.join(self.data_root, 'train')
        else:
            self.data_root = os.path.join(self.data_root, 'test')

        print("Dataset root is " + self.data_root)

        self.transform = transform
        self.samples = self.load_samples()

        print(f"Loaded {len(self.samples)} samples")

    def load_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        all_lines = []
        samples = []
        print(f"Reading data files from {self.data_root}")

        for file_name in os.listdir(self.data_root):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.data_root, file_name)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        all_lines.append(line)

        print("Extracting data from read files...")
        for line in all_lines:
            samples.append(self.extract_sample_from_line(line))

        return samples

    def extract_sample_from_line(self, line):
        all_cols = [int(v) for v in line.strip().split("\t")]

        # Extract label (first column)
        label_data = all_cols[0]
        label_data = np.asarray(label_data,
                                dtype='int16')  # need 2 byte datatype because there are 512 labels
        label_data.reshape(-1, 1)
        label_data = torch.from_numpy(label_data)

        # Extract image data
        image_data = all_cols[1:]
        image_data = np.asarray(image_data).astype('int')

        # Because of Pytorch's channel first convention
        image_data = np.asarray(image_data).reshape(8, 8, -1)

        if self.transform:
            image_data = self.transform(image_data)

        if self.target_transform:
            label_data = self.target_transform(label_data)

        image_data = image_data.float()
        label_data = label_data.float()

        return ((image_data, label_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
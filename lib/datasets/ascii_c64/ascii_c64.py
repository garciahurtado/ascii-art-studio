import os

import pandas as pd

from datasets.multi_dataset import MultiDataset
from datasets.base_dataset import BaseDataset

class AsciiC64(MultiDataset, BaseDataset):
    def __init__(self, transform=None, target_transform=None, train=None, device=None):
        self.data_root = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/data/')
        super().__init__(transform=transform, target_transform=target_transform, train=train, device=device)

    def get_class_counts(self, test=False):
        return super().get_class_counts_from_csv("c64_class_counts")



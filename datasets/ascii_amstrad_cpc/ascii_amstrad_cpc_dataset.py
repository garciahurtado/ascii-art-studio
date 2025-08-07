import os
import pandas as pd
from datasets.multi_dataset import MultiDataset

class AsciiAmstradCPC(MultiDataset):
    dataset_name = "ascii_amstrad_cpc"

    def __init__(self, transform=None, target_transform=None, train=None, device=None):
        super().__init__(transform=transform, target_transform=target_transform, train=train, device=device)

    def get_class_counts(self, test=False):
        if test:
            filename = "amstrad-cpc_class_counts-test"
        else:
            filename = "amstrad-cpc_class_counts"

        return super().get_class_counts_from_csv(filename)
import os
import pandas as pd
from datasets.multi_dataset import MultiDataset


class AsciiAmstradCPC(MultiDataset):
    def __init__(self, transform=None, target_transform=None, train=None, device=None):
        self.data_root = os.path.realpath( os.path.dirname(os.path.realpath(__file__)) + '/data/' )
        super().__init__(transform=transform, target_transform=target_transform, train=train, device=device)

    def get_class_counts(self, test=False):
        ''' Loads the class counts from a CSV file previously generated '''
        if test:
            filename = "amstrad-cpc_class_counts"
        else:
            filename = "amstrad-cpc_class_counts-test"

        filename = os.path.join(self.data_root, f"../{filename}.csv")
        class_counts = pd.read_csv(filename, header=None)
        class_counts = class_counts.to_numpy()

        return class_counts.tolist()

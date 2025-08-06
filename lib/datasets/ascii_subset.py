import os

import pandas as pd
from torch.utils.data import Subset

class AsciiSubset(Subset):
    def __init__(self, dataset, start_index, charset_name, data_root):
        self.charset_name = charset_name
        self.data_root = data_root
        super(AsciiSubset, self).__init__(dataset, start_index)

    def get_class_counts(self):
        return self.get_class_counts_from_csv(self.charset_name + "_class_counts")

    def get_class_counts_from_csv(self, filename):
        ''' Loads the class counts from a CSV file previously generated '''

        filename = os.path.join(self.data_root, f"../{filename}.csv")
        class_counts = pd.read_csv(filename, header=None)
        class_counts = class_counts.to_numpy()

        return class_counts.tolist()
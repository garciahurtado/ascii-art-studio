import os

import pandas as pd


class BaseDataset:
    def get_class_counts_from_csv(self, filename):
        ''' Loads the class counts from a CSV file previously generated '''


        filename = os.path.join(self.data_root, f"../{filename}.csv")
        class_counts = pd.read_csv(filename, header=None)
        class_counts = class_counts.to_numpy()

        return class_counts.tolist()

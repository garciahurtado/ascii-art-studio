import os

import yaml
from torch.utils.data import Dataset
from const import DATASETS_ROOT


class AsciiDataset(Dataset):
    """
    Base class for Ascii datasets. Provides metadata and a default implementation for the get_version() method
    """
    version = "0.0.0"  # this will tell us when a dataset does not have a specific version set
    dataset_name: str  # name of the dataset (ie: ascii_c64, ascii_amstrad_cpc, etc.)
    dataset_path: str  # path to the root directory of the dataset, the one that contains the child AsciiDataset class
    metadata = {}
    metadata_filename = 'metadata.yaml'
    metadata_path: str  # full path to the metadata.yaml file
    data_root: str

    def __init__(self, dataset_name, data_root=None):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(DATASETS_ROOT, dataset_name)
        self.metadata_path = os.path.join(self.dataset_path, self.metadata_filename)

        if not data_root:
            data_root = os.path.join(self.dataset_path, 'processed')

        self.data_root = os.path.realpath(data_root)
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}. Please create it and try again.")

        super(AsciiDataset, self).__init__()

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        if not metadata:
            metadata = {}
        self.metadata = metadata

        print(f"Loaded metadata from {self.metadata_path}")

    def save_metadata(self, metadata=None):
        if metadata:
            self.metadata = metadata

        with open(self.metadata_path, 'w') as f:
            yaml.dump(self.metadata, f)

        print(f"Saved metadata to {self.metadata_path}")

    def get_version(self):
        if not self.metadata:
            self.load_metadata()

        return self.metadata['version']

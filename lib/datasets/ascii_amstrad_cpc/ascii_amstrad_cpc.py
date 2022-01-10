import os

from datasets.multi_dataset import MultiDataset


class AsciiAmstradCPC(MultiDataset):
    def __init__(self, transform=None, target_transform=None, train=None, device=None):
        self.data_root = os.path.realpath( os.path.dirname(os.path.realpath(__file__)) + '/data/' )
        super().__init__(transform=transform, target_transform=target_transform, train=train, device=device)

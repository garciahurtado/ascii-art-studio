from torch.utils.data import Dataset, DataLoader
import torch


# Custom dataset class
class ResampledDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (Tensor): A tensor containing the features of the resampled dataset.
            labels (Tensor): A tensor containing the labels of the resampled dataset.
        """
        assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of samples"

        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, classes_to_exclude):
        self.data = []
        self.labels = []

        for data, label in original_dataset:
            # Check if the label corresponds to an excluded class
            if not any(label[classes_to_exclude]):
                self.data.append(data)
                self.labels.append(label)

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
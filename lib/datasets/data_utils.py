import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Subset

class OneHot:
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def __call__(self, labels):
        """ Encode the labels with one-hot encoding """
        return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.num_labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_class_counts(dataset, num_classes):
    """ Returns an array of class counts present in the dataset. Used for balancing
    datasets by providing weights to the trainer"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_classes,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    counts = None

    for [_, labels] in dataloader:
        labels = torch.argmax(labels, dim=1, keepdim=True)
        curr_counts = np.bincount(labels.flatten(), minlength=num_classes)

        if counts is not None:
            counts = np.add(counts, curr_counts)
        else:
            counts = curr_counts

    return counts


def create_class_weights(labels, mu=0.15):
    labels = {item[0]: item[1] for item in labels}
    total = np.sum(list(labels.values()))
    keys = labels.keys()
    class_weight = dict()

    for key in keys:
        if (labels[key] > 0):
            score = math.log(mu * total / float(labels[key]))
        else:
            score = 0

        class_weight[key] = score if score > 1.0 else 1.0

    return list(class_weight.values())


def split_dataset(dataset, test_split=0.2, random_state=0):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=random_state)
    trainset = Subset(dataset, train_idx)
    testset = Subset(dataset, test_idx)
    return trainset, testset


def get_dataset(train=True, dataset_type=None, num_labels=None):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    target_transform = transforms.Compose(
        [OneHot(num_labels)])

    dataset = dataset_type(
        transform=transform,
        target_transform=target_transform,
        train=train,
        device=get_device())

    return dataset


def write_dataset_class_counts(path, num_classes, dataset_type):
    # Extract class counts for class weights

    print("Evaluating class counts... This may take some time")
    counts = get_class_counts(get_dataset(num_labels=num_classes, dataset_type=dataset_type), num_classes)
    pd.DataFrame(counts).to_csv(f"{path}.csv", header=False)

    # counts = data_utils.get_class_counts(get_dataset(train=False))
    # pd.DataFrame(counts).to_csv(f"{path}-test.csv", header=False)


def seed_init_fn(x):
    seed = 12345 + x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA driver")
    else:
        device = torch.device('cpu')
        print("No CUDA driver available. Using CPU fallback")

    return device


def calculate_padding(self, in_height, in_width, filter_height, filter_width, stride_1, stride2):
    out_height = np.ceil(float(in_height) / float(stride_1))
    out_width = np.ceil(float(in_width) / float(stride2))
    print(f"Out size: {out_height}x{out_width}")

    # The total padding applied along the height and width is computed as:

    if (in_height % stride_1 == 0):
        pad_along_height = max(filter_height - stride_1, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_1), 0)
    if (in_width % stride2 == 0):
        pad_along_width = max(filter_width - stride2, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride2), 0)

    print(pad_along_height, pad_along_width)

    # Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    print(f"Final padding (l,r,t,b): {pad_left}, {pad_right}, {pad_top}, {pad_bottom}")

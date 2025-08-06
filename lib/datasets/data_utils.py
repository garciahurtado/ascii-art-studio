import math
import os
import random
from typing import Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from const import INK_BLUE
from datasets.ascii_subset import AsciiSubset
from datasets.data_augment import AugmentedAsciiDataset
from datasets.multi_dataset import MultiDataset
from debugger import printc


class OneHot:
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def __call__(self, labels):
        """ Encode the labels with one-hot encoding """
        return torch.nn.functional.one_hot(labels.to(torch.float32), num_classes=self.num_labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_class_counts(dataset, num_classes):
    """ Returns an array of class counts present in the dataset. Used for balancing
    datasets by providing weights to the trainer"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    counts = None

    for [_, labels] in dataloader:
        curr_counts = np.bincount(labels, minlength=num_classes)

        if counts is not None:
            counts = np.add(counts, curr_counts)
        else:
            counts = curr_counts

    return counts


def create_class_weights(class_counts, mu=0.001):
    """ Calculates class weights based on the number of samples in each class,
    and the total number of samples in the dataset. The class weights are later used
    to balance the loss function during training. """
    class_counts = {item[0]: item[1] for item in class_counts}
    total = np.sum(list(class_counts.values()))
    labels = class_counts.keys()
    class_weights = dict()

    for label in labels:
        if (class_counts[label] > 0):
            score = math.log(mu * total / float(class_counts[label]))
        else:
            score = 0

        class_weights[label] = score if score > 0 else 0

    # normalize the weights
    # weight_sum = sum(class_weights)
    # class_weights = [w / weight_sum for w in class_weights]

    return list(class_weights)


def _calculate_class_weights(class_counts):
    # DEPRECATED?

    # Calculate class frequencies
    class_counts = torch.tensor(class_counts)
    # total_count = class_counts.sum()
    # class_freqs = class_counts / total_count
    #
    # # Calculate class imbalance ratio
    # rarest_class_freq = class_freqs.min() + 0.0001
    # most_common_class_freq = class_freqs.max()
    # class_imbalance_ratio = rarest_class_freq / most_common_class_freq

    # Calculate effective number of samples
    # effective_num_samples = (1 - class_imbalance_ratio) / class_imbalance_ratio

    # Calculate alpha
    alpha = 0.99
    # effective_num = (1 - np.power(alpha, class_counts)) / (1 - alpha)
    effective_num = (1 - torch.pow(alpha, class_counts)) / (1 - alpha)

    # Calculate mu
    #mu = 1 / effective_num
   # mu_normalized = mu / np.sum(mu)  # This is optional, based on your needs

    #print(f"Class weights: Ideal value of mu: {mu.item():.2f}")

    class_weights = 1 / effective_num
    class_weights = class_weights / torch.sum(class_weights) * len(class_counts)  # Normalize weights

    # Apply class weighting
    # class_weights = 1 / torch.tensor([effective_num]) ** mu
    return class_weights

def write_dataset_class_counts(path, num_classes, dataset_class, char_width=8, char_height=8):
    """ Extracts class counts from a dataset and writes them to a CSV file, to either be analyzed, or to be used for
    balancing the dataset during training. """

    print("Evaluating class counts... This may take some time")

    outfile = f"{path}.csv"
    outdir = os.path.dirname(outfile)
    if not os.access(outdir, os.W_OK):
        raise Exception(f"Cannot write to {outfile}. Permission denied")

    dataset = get_dataset(num_labels=num_classes, dataset_class=dataset_class, char_width=char_width, char_height=char_height)
    counts = get_class_counts(dataset, num_classes)
    pd.DataFrame(counts).to_csv(outfile, header=False)

    print(f"FINISHED: Class counts written to {outfile}")

    # To get the trainset counts only
    # counts = data_utils.get_class_counts(get_dataset(train=False))
    # pd.DataFrame(counts).to_csv(f"{path}-test.csv", header=False)

def get_dataset_details(dataset):
    """
    Generate a comprehensive report about the dataset's properties and statistics.
    
    Args:
        dataset: A PyTorch Dataset object to analyze
        
    Returns:
        dict: A dictionary containing detailed information about the dataset
    """
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
        raise ValueError("Input must be a valid PyTorch Dataset")
    
    details = {
        'basic_info': {},
        'class_distribution': {},
        'data_statistics': {},
        'shape_info': {}
    }
    
    # Basic dataset information
    details['basic_info']['num_samples'] = len(dataset)
    details['basic_info']['dataset_class'] = type(dataset).__name__
    
    # Get a sample to determine data shapes and types
    try:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            # Common case: (data, target) tuple
            data, target = sample[0], sample[1]
            details['shape_info']['input_shape'] = str(tuple(data.shape) if hasattr(data, 'shape') else 'unknown')
            details['shape_info']['target_shape'] = str(tuple(target.shape) if hasattr(target, 'shape') else 'unknown')
            details['shape_info']['input_dtype'] = str(data.dtype) if hasattr(data, 'dtype') else type(data).__name__
            details['shape_info']['target_dtype'] = str(target.dtype) if hasattr(target, 'dtype') else type(target).__name__
        else:
            details['shape_info']['sample_type'] = type(sample).__name__
    except Exception as e:
        details['shape_info']['error'] = f"Could not get sample: {str(e)}"
    
    # Get class distribution if possible
    if hasattr(dataset, 'get_class_counts'):
        try:
            class_counts = dataset.get_class_counts()
            class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)  # Convert to tensor for calculations
            details['class_distribution']['num_classes'] = len(class_counts)
            details['class_distribution']['samples_per_class'] = class_counts
            details['class_distribution']['min_samples'] = int(class_counts_tensor.min())
            details['class_distribution']['max_samples'] = int(class_counts_tensor.max())
            details['class_distribution']['mean_samples'] = float(class_counts_tensor.mean())
            details['class_distribution']['std_samples'] = float(class_counts_tensor.std())
            
            # Calculate class imbalance ratio
            min_samples = details['class_distribution']['min_samples']
            if min_samples > 0:
                imbalance_ratio = details['class_distribution']['max_samples'] / min_samples
                details['class_distribution']['imbalance_ratio'] = float(imbalance_ratio)
        except Exception as e:
            details['class_distribution']['error'] = f"Could not compute class distribution: {str(e)}"
    
    # Additional dataset-specific properties
    for attr in ['char_width', 'char_height', 'num_labels']:
        if hasattr(dataset, attr):
            details['basic_info'][attr] = getattr(dataset, attr)
    
    return details

def print_dataset_details(dataset):
    """
    Print a formatted summary of the dataset's properties and statistics.
    
    Args:
        dataset: A PyTorch Dataset object to analyze and print details for
    """
    details = get_dataset_details(dataset)
    
    # Print basic information
    print("\n" + "="*50)
    print(f"{'DATASET SUMMARY':^50}")
    print("="*50)
    
    # Print basic info
    print("\nBasic Information:")
    print("-" * 50)
    for key, value in details['basic_info'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Print shape info if available
    if details['shape_info']:
        print("\nShape Information:")
        print("-" * 50)
        for key, value in details['shape_info'].items():
            if not key.startswith('error'):
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Print class distribution if available
    if 'class_distribution' in details and details['class_distribution']:
        dist = details['class_distribution']
        print("\nClass Distribution:")
        print("-" * 50)
        if 'error' in dist:
            print(f"Error: {dist['error']}")
        else:
            print(f"Number of classes: {dist.get('num_classes', 'N/A')}")
            print(f"Samples per class: {dist.get('min_samples', 'N/A')} (min) - {dist.get('max_samples', 'N/A')} (max)")
            if 'imbalance_ratio' in dist:
                print(f"Class imbalance ratio: {dist['imbalance_ratio']:.1f}x")
    
    print("\n" + "="*50 + "\n")
    return details

def split_dataset(dataset, test_split=0.2, random_state=0, charset_name=None):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=random_state)
    trainset = AsciiSubset(dataset, train_idx, charset_name, dataset.data_root)
    testset = AsciiSubset(dataset, test_idx, charset_name, dataset.data_root)
    return trainset, testset

def get_dataset(train=True, dataset_class: Type[MultiDataset] = None, char_height=8, char_width=8, num_labels=None, augment=False, augment_params: dict = None):
    # Skip the OneHot transform as long as we are using CrossEntropyLoss
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])

    # target_transform = transforms.Compose(
    #     [OneHot(num_labels)])

    dataset = dataset_class(
        train=train,
        device=get_device())

    dataset.char_width = char_width
    dataset.char_height = char_height

    # Conditionally wrap the dataset with the augmentation class
    if train and augment:
        print("Data augmentation enabled for the training set.")
        dataset = AugmentedAsciiDataset(
            original_dataset=dataset,
            is_train=True,
            augment_params=augment_params)

    print(f"Dataset: {dataset_class.__name__} loaded (Train: {train})")
    print("-" * 50)
    print_dataset_details(dataset)
    print("-" * 50)

    return dataset

def seed_init_fn(x):
    seed = 12345 + x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        printc(f"Using CUDA driver version: {torch.version.cuda}", INK_BLUE)
    else:
        device = torch.device('cpu')
        printc("No CUDA available. Using CPU fallback")

    return device

def concat_files(path, ext):
    all_text = ""

    for filename in os.listdir(path):
        # Check if the file is a .txt file
        if filename.endswith(ext):
            # Open the file in read mode
            with open(os.path.join(path, filename), 'r') as file:
                file_contents = file.read()
                all_text += file_contents + '\n'

    # Write the concatenated text to a summary file
    with open(os.path.join(path, 'all-files.txt'), 'w') as summary_file:
        summary_file.write(all_text)

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

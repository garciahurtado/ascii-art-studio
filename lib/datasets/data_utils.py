import math
import random

import numpy as np
import torch


class OneHot:
    def __call__(self, labels):
        """ Encode the labels with one-hot encoding """
        num_labels = 486
        return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=num_labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_class_counts(dataset):
    """ Returns an array of class counts present in the dataset. Used for balancing
    datasets by providing weights to the trainer"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=486,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    counts = None

    for [_, labels] in dataloader:
        labels = torch.argmax(labels, dim=1, keepdim=True)
        curr_counts = np.bincount(labels.flatten(), minlength=486)

        if counts is not None:
            counts = np.add(counts, curr_counts)
        else:
            counts = curr_counts

    return counts

def create_class_weights(labels, mu=0.15):
    labels = {item[0]:item[1] for item in labels}
    total = np.sum(list(labels.values()))
    keys = labels.keys()
    class_weight = dict()

    for key in keys:
        if(labels[key] > 0):
            score = math.log(mu * total / float(labels[key]))
        else:
            score = 0

        class_weight[key] = score if score > 1.0 else 1.0

    return list(class_weight.values())

def seed_init_fn(x):
   seed = 12345 + x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return


import torch


class OneHot:
    def __call__(self, labels):
        """ Encode the labels with one-hot encoding """
        num_labels = 512
        return torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=num_labels)

    def __repr__(self):
        return self.__class__.__name__ + '()'


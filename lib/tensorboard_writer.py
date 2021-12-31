import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TensorboardWriter(SummaryWriter):
    """ Extends the base SummaryWriter for TensorBoard with the ability to automatically
    name the log directory based on the specific dataset used"""
    def __init__(self, dir, dataset):
        dataset_type = type(dataset)
        now = datetime.now()
        logdir = dir + dataset_type.__name__ + "/" + now.strftime("%b%d_%H-%M-%S")

        print("Writing Tensorboard logs to: " + logdir)
        super().__init__(logdir)

    def calc_accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label
        @ref https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002

        :param model:
        :param x:
        :param y:
        :return:
        """
        # get the scores for each class (or logits)
        # y_logits = model(x)  # unnormalized probs

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = model(x).max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = x.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == y).sum().item() / n
        return acc
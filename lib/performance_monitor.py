import numpy as np
import torch

class PerformanceMonitor:
    """Maintains the prediction performance state of several variables in order
    to facilitate calculating training stats like avg. loss and accuracy"""

    def __init__(self):
        self.total_predictions = 0
        self.correct_predictions = 0
        self.loss = []
        self.loss_window = 100 # Avg Loss is calculated based on the last N samples

    def reset(self):
        self.total_predictions = 0
        self.correct_predictions = 0

    def add_predictions(self, results, labels):
        self.total_predictions += labels.size(0)
        _, predicted = torch.max(results.data, 1)
        predicted = torch.reshape(predicted, [labels.size(0),1])

        correct = (predicted == labels).sum().item()
        self.correct_predictions += correct

    def get_accuracy(self):
        if not self.total_predictions:
            return 0

        accuracy = 100 * (self.correct_predictions / self.total_predictions)
        return accuracy

    def get_avg_loss(self):
        samples = np.array(self.loss)

        num_samples = len(samples) if (len(samples) < self.loss_window) else self.loss_window

        if num_samples == 1:
            return samples[0]

        end = len(samples)
        avg_loss = np.average(samples[end - num_samples:end])
        return avg_loss
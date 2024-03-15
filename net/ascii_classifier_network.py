import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiClassifierNetwork(nn.Module):
    def __init__(self, num_labels=None):
        super(AsciiClassifierNetwork, self).__init__()

        # Architecture
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding="same", stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)


        self.fc2 = nn.Linear(4096, num_labels)
        self.fc2_norm = nn.BatchNorm1d(num_labels)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension (from 2D (x/y) to 1D tensors)
        x = self.fc2(x)
        x = self.fc2_norm(x)
        output = F.log_softmax(x, dim=1)

        return output

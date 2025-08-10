import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiC64Network(nn.Module):
    """ Try to keep this class as simple as possible, so that it will be easy to serialize and deserialize when we
    save it alongside the model weights """
    version = '0.0.5'

    def __init__(self, num_labels):
        super(AsciiC64Network, self).__init__()

        # Input: 1x8x8 (channels x height x width)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=1)
        self.conv1_norm = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(288, 512)
        self.fc1_norm = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, num_labels)
        self.fc2_norm = nn.BatchNorm1d(num_labels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_norm(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension (from 2D (x/y) to 1D tensors)

        x = self.fc1(x)
        x = self.fc1_norm(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc2_norm(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)

        return output

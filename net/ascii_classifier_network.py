import torch
import torch.nn as nn
import torch.nn.functional as F


class AsciiClassifierNetwork(nn.Module):
    def __init__(self, num_labels=None):
        super(AsciiClassifierNetwork, self).__init__()

        # Architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding="same", stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same", stride=1)

        # Assuming the input is 8x8, after two pooling layers it will be 2x2
        self.fc1 = nn.Linear(32 * 2 * 2, 64)  # 32 channels, 2x2 images
        self.fc1_norm = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = self.fc1(x)
        x = self.fc1_norm(x)
        x = F.relu(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

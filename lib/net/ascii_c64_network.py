import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiC64Network(nn.Module):
    """ Try to keep this class as simple as possible, so that it will be easy to serialize and deserialize when we
    save it alongside the model weights """
    version = '0.0.3'

    def __init__(self, num_labels):
        super(AsciiC64Network, self).__init__()

        # Input: 1x8x8 (channels x height x width)
        
        # First conv layer: 1x8x8 -> 64x8x8 (same spatial size due to padding=1)
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
        # Second conv layer: 64x8x8 -> 64x5x5
        # (8 + 2*padding - kernel_size) // stride + 1 = (8 + 4 - 3) // 2 + 1 = 5
        # self.conv2 = nn.Conv2d(
        #     32,
        #     64,
        #     kernel_size=3,
        #     padding=2,
        #     stride=2
        # )

        # Calculate the size for the first fully connected layer
        # After conv2: 64 channels * 5 * 5 = 1600
        self.fc1 = nn.Linear(2048, 256)
        self.fc1_norm = nn.BatchNorm1d(256)

        # Final output layer
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        # Convolution and pooling layers
        x = F.relu(self.conv1(x))  # 1x8x8 -> 64x8x8
        # x = F.relu(self.conv2(x))  # 64x8x8 -> 64x5x5

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)  # 64x5x5 -> 1600

        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1(x)
        x = self.fc1_norm(x)
        x = F.relu(x)

        # Output layer
        x = self.fc2(x)

        # Applying log_softmax to get log-probabilities
        output = F.log_softmax(x, dim=1)

        return output

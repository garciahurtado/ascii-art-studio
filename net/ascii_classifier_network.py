import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiClassifierNetwork(nn.Module):
    def __init__(self, num_labels):
        super(AsciiClassifierNetwork, self).__init__()

        # Assuming input images are single-channel (grayscale) of size 8x8
        # First conv layer will output 64 channels, same spatial size due to padding
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1)

        # Second conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)

        self.fc1 = nn.Linear(8192, 512)
        self.fc1_norm = nn.BatchNorm1d(512)
        #self.dropout1 = nn.Dropout(0.5)

        # Final output layer
        self.fc2 = nn.Linear(512, num_labels)

    def forward(self, x):
        # Convolution and pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1_norm(self.fc1(x))
        x = F.relu(x)
        #x = self.dropout1(x)

        # No need for an activation function here because
        # F.log_softmax will be applied afterwards
        x = self.fc2(x)

        # Applying log_softmax to get log-probabilities
        output = F.log_softmax(x, dim=1)

        return output

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiClassifierNetwork(nn.Module):
    def __init__(self, num_labels=None):
        super(AsciiClassifierNetwork, self).__init__()

        # Hyperparameters

        # Architecture
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1, stride=2)
        self.conv1_norm = nn.BatchNorm2d(128)
        # self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1)

        # self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.conv2_norm = nn.BatchNorm2d(32)

        # self.fc0 = nn.Conv2d(1, 64, kernel_size=6, padding=1)
        # self.fc0_norm = nn.BatchNorm1d(2048)

        self.fc1 = nn.Linear(2048, 512)
        self.fc1_norm = nn.BatchNorm1d(512)

        # self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_labels)
        self.fc2_norm = nn.BatchNorm1d(num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_norm(x)
        x = F.relu(x)

        # x = self.conv2(x)
        # x = self.conv2_norm(x)
        # x = F.relu(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension (from 2D to 1D tensors)
        x = self.fc1(x)
        x = self.fc1_norm(x)
        x = F.relu(x)
        #x = self.dropout(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
    @classmethod
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
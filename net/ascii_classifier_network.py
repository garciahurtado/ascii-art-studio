import torch
import torch.nn as nn
import torch.nn.functional as F

class AsciiClassifierNetwork(nn.Module):
    def __init__(self):
        super(AsciiClassifierNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3),stride=(2,2),padding=1)
        # self.conv1_norm = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(128, 512)
        self.fc_norm = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)


    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv1_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
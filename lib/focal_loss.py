import torch
from torch import nn
class FocalLoss(nn.Module):

    def __init__(self, alpha=5.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets,
                                                    reduction='none')  # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss
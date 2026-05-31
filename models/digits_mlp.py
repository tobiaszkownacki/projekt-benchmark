from torch import flatten
import torch.nn as nn
import torch.nn.functional as F


class DigitsMLP(nn.Module):
    def __init__(self, in_channels=1, img_size=8, num_classes=10):
        super().__init__()
        input_features = in_channels * img_size * img_size

        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

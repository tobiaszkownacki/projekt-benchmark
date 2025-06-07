import torch.nn as nn


class HeartDisease(nn.Module):
    def __init__(self, input_size=13, output_size=5):
        super(HeartDisease, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.fc(x)

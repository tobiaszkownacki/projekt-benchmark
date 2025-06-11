import torch.nn as nn


class WineQuality(nn.Module):
    def __init__(self, input_size=12, output_size=2):
        super(WineQuality, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )

    def forward(self, x):
        return self.fc(x)
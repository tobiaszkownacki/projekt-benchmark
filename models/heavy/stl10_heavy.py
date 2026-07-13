import torch
import torch.nn as nn

class STL10Heavy(nn.Module):
    def __init__(self, input_channels=3, output_size=10, channels=(32, 64, 128)):
        super().__init__()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
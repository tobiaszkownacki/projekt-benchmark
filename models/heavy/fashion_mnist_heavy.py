import torch 
import torch.nn as nn

class FashionMNISTHeavy(nn.Module):
    def __init__(self, input_channels=1, output_size=10):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
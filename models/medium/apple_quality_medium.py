import torch
import torch.nn as nn

class AppleQualityMedium(nn.Module):
    def __init__(self, input_size=7, output_size=2):
        super().__init__()
    
    def forward(self, x):
        return x
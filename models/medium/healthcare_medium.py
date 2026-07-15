import torch 
import torch.nn as nn

class HealthcareMedium(nn.Module):
    def __init__(self, input_size=31, output_size=3):
        super().__init__()
    
    def forward(self, x):
        return x
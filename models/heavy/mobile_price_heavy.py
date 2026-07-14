import torch.nn as nn

class MobilePriceHeavy(nn.Module):
    def __init__(self, input_size=20, output_size=4):
        super().__init__()
    
    def forward(self, x):
        return x
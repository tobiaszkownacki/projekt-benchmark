import torch.nn as nn

class CreditScoreLight(nn.Module):
    def __init__(self, input_size=23, output_size=3):
        super().__init__()
            
    def forward(self, x):
        return x
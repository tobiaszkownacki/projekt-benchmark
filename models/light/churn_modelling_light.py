import torch.nn as nn

class ChurnModellingLight(nn.Module):
    def __init__(self, input_size=12, output_size=2):
        super().__init__()
                    
    def forward(self, x):
        return x
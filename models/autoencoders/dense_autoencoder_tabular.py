import torch.nn as nn


class DenseAutoEncoderTabular(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super().__init__()
        input_dim = input_dim[0]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

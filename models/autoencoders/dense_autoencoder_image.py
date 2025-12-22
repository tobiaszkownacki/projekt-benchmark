import torch.nn as nn
import numpy as np


class DenseAutoEncoderImage(nn.Module):
    def __init__(self, input_shape, latent_dim=64):
        """
        input_shape: krotka wymiarów, np. (3, 32, 32) dla CIFAR lub (1, 8, 8) dla Digits
        """
        super().__init__()

        self.input_shape = input_shape
        self.flat_dim = np.prod(input_shape)

        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # 3. Dekoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)

        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)

        return decoded.view(x.size(0), *self.input_shape)

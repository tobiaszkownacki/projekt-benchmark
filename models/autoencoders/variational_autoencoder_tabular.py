from torch import nn
import torch
import torch.nn.functional as F


class VariationalAutoEncoderTabular(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=2):
        super().__init__()
        input_dim = input_dim[0]

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.fc_decode_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decode_2 = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)

        z = self.reparameterize(mu, logvar)

        h_dec = F.relu(self.fc_decode_1(z))
        recon_x = self.fc_decode_2(h_dec)

        return recon_x, mu, logvar

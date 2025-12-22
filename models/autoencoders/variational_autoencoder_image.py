from torch import nn
import torch


class VariationalAutoEncoderImage(nn.Module):
    def __init__(self, num_channels=3, latent_dim=128):
        super().__init__()

        # Enkoder Conv
        self.enc_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.flatten_dim = 128 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.enc_conv(x)
        x_flat = x.view(x.size(0), -1)

        mu = self.fc_mu(x_flat)
        logvar = self.fc_var(x_flat)

        z = self.reparameterize(mu, logvar)

        z = self.decoder_input(z)
        z_reshaped = z.view(z.size(0), 128, 4, 4)
        recon_x = self.dec_conv(z_reshaped)

        return recon_x, mu, logvar

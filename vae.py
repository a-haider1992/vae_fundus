import torch
import torch.nn as nn
import torchvision.models as models

class VAE_1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_1, self).__init__()
        self.input_dim = input_dim # shape: (channels, width, height)
        self.latent_dim = latent_dim # shape: int
        self.encoder = models.resnet50(pretrained=True)

        # Freeze all the layers in the pretrained model
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(512, 2 * latent_dim)  # 2 * latent_dim outputs for mean and variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            # nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * self.input_dim[1] * self.input_dim[2]),  # Output size matches input size (3, 2100, 2100)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)  # Split into mean and log variance
        z = self.reparameterize(mu, logvar)

        # Decoder
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, 3, self.input_dim[1], self.input_dim[2])  # Reshape to match input size
        return x_recon, mu, logvar


import torch
import torch.nn as nn
from vae import VAE_1
from clustering import KMeans

class AutoencoderKMeans(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoencoderKMeans, self).__init__()
        self.autoencoder = VAE_1(input_dim, latent_dim)
    
    def forward(self, x):
        encoded, recon_x, mu, logvar = self.autoencoder(x)
        return encoded, recon_x, mu, logvar
    
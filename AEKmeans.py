import torch
import torch.nn as nn
from vae import VAE_1
from clustering import KMeans

class AutoencoderKMeans(nn.Module):
    def __init__(self, input_dim, latent_dim, num_clusters):
        super(AutoencoderKMeans, self).__init__()
        self.autoencoder = VAE_1(input_dim, latent_dim)
        self.kmeans = KMeans(num_clusters, latent_dim)
    
    def forward(self, x):
        encoded, recon_x, mu, logvar = self.autoencoder(x)
        assignments = self.kmeans(encoded)
        return recon_x, mu, logvar,assignments
    
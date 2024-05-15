import torch
import torch.nn as nn
from piqa import SSIM

class SSIMLoss(SSIM):
        def forward(self, x, y):
            return 1. - super().forward(x, y)

class Loss():
    def __init__(self, dimen, device):
        self.dimen = dimen
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.ssim_loss = SSIMLoss().to(device)

    def loss_function(self, recon_x, x, encoded, centroids, assignments, mu=None, logvar=None):
        ssim_loss = self.ssim_loss(recon_x, x)
        recon_x = recon_x.view(-1, 3 * self.dimen * self.dimen)
        x = x.view(-1, 3 * self.dimen * self.dimen)
        reconstruction_loss = self.mse_loss(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        regularization_loss = torch.mean(torch.norm(encoded - centroids[assignments], dim=1))
        return reconstruction_loss + kl_divergence + ssim_loss + regularization_loss, ssim_loss

        
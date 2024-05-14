import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import CustomImageDataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pdb

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256*7*7, latent_dim)
        self.fc_logvar = nn.Linear(256*7*7, latent_dim)
        
        self.decoder = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, 448, kernel_size=4, stride=1, padding=0),  # Adjusted for desired output size
    nn.ReLU(),
    nn.ConvTranspose2d(448, 224, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(224, 112, kernel_size=4, stride=2, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(112, 56, kernel_size=4, stride=2, padding=2),
    nn.ReLU(),
    nn.ConvTranspose2d(56, 28, kernel_size=4, stride=2, padding=2),
    nn.ReLU(),
    nn.ConvTranspose2d(28, input_dim[0], kernel_size=4, stride=2, padding=3),
    nn.Sigmoid()
)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256*7*7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = z.view(-1, latent_dim, 1, 1)
        x = self.decoder(z)
        return x, mu, logvar

# Initialize hyperparameters
input_dim = (3, 2100, 2100)  # RGB images
latent_dim = 32
learning_rate = 1e-4
batch_size = 32
epochs = 100

# Initialize TensorBoard writer
writer = SummaryWriter()

# Load your custom dataset using ImageFolder
transform = transforms.Compose([
    transforms.Resize((2100, 2100)),
    transforms.ToTensor(),
])

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# pdb.set_trace()
dataset = CustomImageDataset(txt_file="file_paths.txt",root_dir="convex", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the VAE model
vae = VAE(input_dim, latent_dim)

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# Use DataParallel to utilize multiple GPUs if available
if torch.cuda.device_count() > 1:
    vae = nn.DataParallel(vae)

# Training loop
vae.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        if data.shape == recon_batch.shape:
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item() / len(data)))
        else:
            print(f'Input batch shape: {data.shape}')
            print(f'Reconstructed batch shape: {recon_batch.shape}')
            raise Exception("Shape of input and reconstructed input does not match!!")
        writer.add_scalar('Loss/train_batch', loss.item() / len(data), epoch * len(dataloader) + batch_idx)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(dataloader.dataset)))
    # writer.add_scalar('Loss/train', total_loss / len(dataloader.dataset), epoch)

# Save the trained model
torch.save(vae.state_dict(), 'vae_model.pth')
# Close TensorBoard writer
writer.close()

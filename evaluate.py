from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import pdb

class VAE_1(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_1, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = models.resnet50(pretrained=True)

        # Freeze all the layers in the pretrained model
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

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
            nn.Linear(2048, 3 * 128 * 128),  # Output size matches input size (3, 2100, 2100)
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
        x_recon = x_recon.view(-1, 3, 128, 128)  # Reshape to match input size
        return x_recon, mu, logvar
    
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use DataParallel to utilize multiple GPUs if available
# Load the trained model
latent_dim = 300
input_dim = (3, 128, 128)
pdb.set_trace()
trained_model = VAE_1(latent_dim)
trained_model.load_state_dict(torch.load('vae_model_1.pth'))
trained_model.eval()
if torch.cuda.device_count() > 1:
    vae = torch.nn.DataParallel(trained_model)

# Load a test image
test_image_path = "image7.jpg"  # Replace with the path to the user-provided test image

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([input_dim[1], input_dim[2]]),
    transforms.ToTensor(),
    ])

test_image = Image.open(test_image_path)
test_image = transform(test_image).unsqueeze(0)  # Apply the same transform as the dataset and add batch dimension

test_image = test_image.to(device)
# Reconstruct the test image using the trained model
with torch.no_grad():
    reconstructed_image, _, _ = trained_model(test_image)
# Display the original and reconstructed images

fig, axes = plt.subplots(1, 2)
axes[0].imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

plt.show()
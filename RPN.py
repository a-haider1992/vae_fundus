import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import pdb
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import transforms

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def RPN(image):
    '''
    image: cv2 image
    return: object_patches, boxes, labels
    '''
    def extract_object_patches(image, boxes):
        patches = []
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            patch = image[x_min: x_max, y_min: y_max]
            patches.append(patch)
        return patches
    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Transform the image
    input_image = transform(image).unsqueeze(0)

    # Get predictions from the model
    with torch.no_grad():
        predictions = model(input_image)

    boxes = predictions[0]['boxes'].tolist()
    labels = predictions[0]['labels'].tolist()

    # pdb.set_trace()

    object_patches = extract_object_patches(image, boxes)
    # print(f"Found {len(object_patches)} objects in the image")

    # for i, patch in enumerate(object_patches):
    #     cv2.imwrite(f"object_patch_{i}.jpg", patch)

    return object_patches, boxes, labels


class VAE(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
        )
        
        # self.fc = nn.Linear(1024*2*2, latent_dim)
        # Calculate the dimension after the convolutional layers
        # conv_output_size = self._get_conv_output(input_dim)
        
        # Fully connected layers for mean and variance
        self.fc_mu = nn.Linear(1024*2*2, latent_dim)
        self.fc_logvar = nn.Linear(1024*2*2, latent_dim)
        
        # Decoder
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_dim[0], kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(4, input_dim[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = x.view(-1, 1024*2*2)
        mu = self.fc_mu(x)
        var = self.fc_logvar(x)
        z = mu.view(-1, self.latent_dim, 1, 1)
        x = self.decoder(z)
        return x, mu, var


from vae import VAE_1
from AEKmeans import AutoencoderKMeans

import time
import logging
import matplotlib.pyplot as plt
from PIL import Image
import optuna
from piqa import SSIM
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import VAEDataset, ExplanationsPatchesDataset
import pdb
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import os
import optuna
import argparse
from loss import Loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='VAE-GMM')
    parser.add_argument('--input_dim', type=int, nargs='+', default=[3, 256, 256], help='input dimensions')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--num_clusters', type=int, default=3, help='number of clusters')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_dim = tuple(args.input_dim)
    learning_rate = args.learning_rate
    epochs = args.epochs
    max_norm = args.max_norm
    num_clusters = args.num_clusters

    summary_writer = SummaryWriter()

    logging.basicConfig(filename='vae.log', level=logging.INFO)
    logging.info(f'Input dimensions: {input_dim}')
    logging.info(f'Learning rate: {learning_rate}')
    logging.info(f'Epochs: {epochs}')
    logging.info(f'Max norm: {max_norm}')
    logging.info(f'Execution started: {time.strftime("%Y-%m-%d-%H-%M-%S")}')

    # Initialize TensorBoard writer
    # writer_res = SummaryWriter()

    # Load your custom dataset using ImageFolder
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([input_dim[1], input_dim[2]]),
        transforms.ToTensor(),
        ])

    def plot_tsne(centroids, assignments, filename='tsne.png', outlier_threshold=1.5):
        # Detach and convert tensors to numpy arrays
        centroids = torch.mean(centroids, dim=0)
        centroids = centroids.detach().cpu().numpy()
        assignments = assignments.detach().cpu().numpy()

        # Get the centroids corresponding to the assignments
        assigned_centroids = centroids[assignments]

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=50.0, random_state=0)
        tsne_data = tsne.fit_transform(assigned_centroids)

        # Calculate the mean and standard deviation of the t-SNE data
        mean = np.mean(tsne_data, axis=0)
        std_dev = np.std(tsne_data, axis=0)

        # Identify the indices of points that are not outliers
        non_outliers = np.all(np.abs(tsne_data - mean) < outlier_threshold * std_dev, axis=1)

        # Filter the t-SNE data and assignments to exclude outliers
        tsne_data_no_outliers = tsne_data[non_outliers]
        assignments_no_outliers = assignments[non_outliers]

        # Plot t-SNE
        plt.scatter(tsne_data_no_outliers[:, 0], tsne_data_no_outliers[:, 1], c=assignments_no_outliers, cmap='viridis')
        plt.title('t-SNE of Assigned Centroids')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(filename)

    # Define the trainable function
    def train_vae(config):
        # Set the batch_size parameter
        batch_size = config["batch_size"]
        latent_dim = config["latent_dim"]
        print(f"Using Batch size: {batch_size}")
        print(f"Using Latent dim: {latent_dim}")
        
        # Update the batch_size in the dataloader
        train_dataset = VAEDataset(txt_file="fundus_train.txt", root_dir=".", transform=transform)
        test_dataset = VAEDataset(txt_file="fundus_test.txt", root_dir=".", transform=transform)

        # train_size = len(train_dataset)
        # test_size = len(test_dataset)

        # train_indices = list(range(train_size))
        # test_indices = list(range(test_size))

        # train_limit = int(train_size * 0.1)
        # test_limit = int(test_size * 0.1)

        # pdb.set_trace()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=None)

        model = AutoencoderKMeans(input_dim, latent_dim, num_clusters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)
        loss_func = Loss(input_dim[1], device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        total_loss = 0.0
        total_ssim = 0.0
        # Training loop
        model.train()
        logging.info("Training the model")
        for epoch in range(epochs):
            for batch_idx, data in enumerate(train_loader):
                data = data[0].to(device)  # Move data to GPU if available
                optimizer.zero_grad()
                # pdb.set_trace()
                encoded, recon_batch, mu, logvar, centroids, assignments = model(data)                
                if data.shape == recon_batch.shape:
                    loss, _ = loss_func.loss_function(recon_batch, data, encoded, centroids, assignments,mu, logvar)
                    loss.backward()
                    total_loss += loss.item()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    
                    if batch_idx % 100 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item() / len(data)))
                        logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item() / len(data)))
                        plot_tsne(centroids, assignments)
                        summary_writer.add_scalar('Loss/train_batch', loss.item() / len(data), batch_idx)
                else:
                    print(f'Input batch shape: {data.shape}')
                    print(f'Reconstructed batch shape: {recon_batch.shape}')
                    raise Exception("Shape of input and reconstructed input does not match!!")
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / (len(train_loader.dataset) * epochs)))
            logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / (len(train_loader.dataset) * epochs)))
            # summary_writer.add_scalar('Loss/train_epoch', total_loss / (len(train_loader.dataset) * epochs), epoch)

        # Evaluate the trained model
        model.eval()
        total_loss = 0.0
        logging.info("Evaluating the model")
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data[0].to(device)  # Move data to GPU if available
                encoded, recon_batch, mu, logvar, centroids, assignments = model(data)
                if data.shape == recon_batch.shape:
                    loss, ssim = loss_func.loss_function(recon_batch, data, encoded, centroids, assignments,mu, logvar)
                    total_loss += loss.item()
                    total_ssim += ssim.item()
                else:
                    print(f'Input batch shape: {data.shape}')
                    print(f'Reconstructed batch shape: {recon_batch.shape}')
                    raise Exception("Shape of input and reconstructed input does not match!!")
        average_loss = total_loss / len(test_loader.dataset)
        average_ssim = total_ssim / len(test_loader.dataset)
        # print(f'====> Test set loss: {average_loss:.4f}')
        # print(f'====> Test set SSIM: {average_ssim:.4f}')
        logging.info(f'====> Test set loss: {average_loss:.4f}')
        logging.info(f'====> Test set SSIM: {average_ssim:.4f}')
        torch.save(model.state_dict(), 'model.pth')
        return average_loss
    
    def infer_vae(config):
        # Set the batch_size parameter
        batch_size = config["batch_size"]
        latent_dim = config["latent_dim"]
        print(f"Using Batch size: {batch_size}")
        print(f"Using Latent dim: {latent_dim}")
        
        # Update the batch_size in the dataloader
        # filename = 'fundus_explanations.txt' contains the list of images to be used for inference
        # The images are explanations generated DeepCover (refer to the paper for more details)
        # The images are the found locations by DeepCover in the original fundus images
        test_dataset = ExplanationsPatchesDataset(txt_file="fundus_explanations.txt", root_dir=".", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=None)

        model = AutoencoderKMeans(input_dim, latent_dim, num_clusters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        logging.info("Evaluation mode: Generating explanations clusters")
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data[0].to(device)
                encoded, recon_batch, mu, logvar, centroids, assignments = model(data)
                plot_tsne(centroids, assignments, filename='tsne_explanations.png')
    
    if not os.path.exists('model.pth'):
        train_vae({"batch_size": 128, "latent_dim": 350})
    else:
        infer_vae({"batch_size": 64, "latent_dim": 350})

    # Define the search space
    # Define the objective function to optimize
    # def objective(trial):
    #     # Define the search space
    #     batch_size = trial.suggest_categorical("batch_size", [64, 128])
    #     latent_dim = trial.suggest_int("latent_dim", 300, 400)
    #     # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        
    #     # Train the VAE model with the specified configuration
    #     config = {"batch_size": batch_size, "latent_dim": latent_dim}
    #     loss = train_vae(config)
    #     return loss

    # # Create an Optuna study
    # study_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    # study = optuna.create_study(storage="sqlite:///db.sqlite3", 
    #                             direction="minimize", study_name=f"Study_{study_name}")


    # # Optimize the objective function
    # study.optimize(objective, n_trials=10)
    # # Save the best performing model
    # best_model = study.best_trial.user_attrs['model']
    # torch.save(best_model.state_dict(), 'best_model.pth')
    # logging.info(f"Best value: {study.best_value} (params: {study.best_params})")
    # logging.info(f'Execution ended: {time.strftime("%Y-%m-%d-%H-%M-%S")}')
    # print(f"Best value: {study.best_value} (params: {study.best_params})")

    
if __name__ == "__main__":
    main()
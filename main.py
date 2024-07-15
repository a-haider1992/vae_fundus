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
import os, shutil
import optuna
import cv2
import argparse
from loss import Loss
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from clustering import KMeans

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

    def plot_tsneV1(encoded, assignments, labels=None, filename='tsne_plot.png'):
        # Perform t-SNE on the encoded vectors
        tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=2000, random_state=0)
        tsne_results = tsne.fit_transform(encoded.detach().cpu().numpy())
        
        # Calculate the centroids of the clusters in the t-SNE space
        num_clusters = len(torch.unique(assignments))
        centroids = np.zeros((num_clusters, 2))

        for i in range(num_clusters):
            centroids[i] = tsne_results[assignments.detach().cpu().numpy() == i].mean(axis=0)
        
        # Plot the t-SNE results
        plt.figure(figsize=(10, 7))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=assignments.detach().cpu().numpy(), cmap='viridis', s=5)
        
        # Plot the centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
        
        plt.title('t-SNE of SNP=CFH')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(label='Cluster Assignment')
        plt.legend()
        plt.savefig(filename)

    # def plot_tsneV1(encoded, assignments, labels=None, filename='tsne_plot.png'):
    #     # Perform t-SNE on the encoded vectors
    #     tsne = TSNE(n_components=2, random_state=0)
    #     tsne_results = tsne.fit_transform(encoded.detach().cpu().numpy())
        
    #     # Calculate the centroids of the clusters in the t-SNE space
    #     num_clusters = len(torch.unique(assignments))
    #     centroids = np.zeros((num_clusters, 2))

    #     for i in range(num_clusters):
    #         centroids[i] = tsne_results[assignments.detach().cpu().numpy() == i].mean(axis=0)
        
    #     # Plot the t-SNE results
    #     plt.figure(figsize=(10, 7))
    #     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=assignments.detach().cpu().numpy(), cmap='viridis', s=5)
            
    #     # Plot the centroids
    #     plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
        
    #     # Annotate each point with its label
    #     if labels is not None:
    #         for i, txt in enumerate(labels):
    #             plt.text(tsne_results[i, 0], tsne_results[i, 1], str(txt), fontsize=9)
        
    #     plt.title('t-SNE of Encoded Vectors')
    #     plt.xlabel('t-SNE Dimension 1')
    #     plt.ylabel('t-SNE Dimension 2')
    #     plt.colorbar(scatter, label='Cluster Assignment')
    #     plt.legend()
    #     plt.savefig(filename)

    # Define the trainable function
    def train_vae(config):
        # Set the batch_size parameter
        batch_size = config["batch_size"]
        latent_dim = config["latent_dim"]
        print(f"Using Batch size: {batch_size}")
        print(f"Using Latent dim: {latent_dim}")
        
        # Update the batch_size in the dataloader
        train_dataset = VAEDataset(txt_file="fundus_patches.txt", root_dir=".", transform=transform)
        # test_dataset = VAEDataset(txt_file="fundus_test.txt", root_dir=".", transform=transform)
        test_dataset = ExplanationsPatchesDataset(txt_file="fundus_explanations.txt", root_dir=".", transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=None)

        model = AutoencoderKMeans(input_dim, latent_dim)
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
        encoded_vectors = []
        assignments_list = []
        kmeans = KMeans(num_clusters, latent_dim).to(device)
        # Training loop
        model.train()
        logging.info("Training the model")
        try:
            for epoch in range(epochs):
                for batch_idx, data in enumerate(train_loader):
                    image, masked_image = data[0], data[1]  # Move data to GPU if available
                    # pdb.set_trace()
                    image, masked_image = image.to(device), masked_image.to(device)
                    # data = data[0].to(device)  # Move data to GPU if available
                    optimizer.zero_grad()
                    # pdb.set_trace()
                    encoded, recon_batch, mu, logvar = model(masked_image)                
                    if image.shape == recon_batch.shape:
                        loss, _ = loss_func.loss_function(recon_batch, image, mu, logvar)
                        loss.backward()
                        total_loss += loss.item()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        optimizer.step()
                        encoded_vectors.append(encoded)
                        centroids, assignments = kmeans(encoded.detach())
                        assignments_list.append(assignments)
            
                        # Update centroids
                        kmeans.update_centroids(encoded.detach(), assignments)
                        if batch_idx % 100 == 0:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.item() / len(data)))
                            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.item() / len(data)))
                            summary_writer.add_scalar('Loss/train_batch', loss.item() / len(data), batch_idx)
                            # encoded_vectors1 = torch.cat(encoded_vectors)
                            # assignments_list1 = torch.cat(assignments_list)
                            # plot_tsneV1(encoded_vectors1, assignments_list1, filename='tsne_plot.png')
                    else:
                        print(f'Input batch shape: {data.shape}')
                        print(f'Reconstructed batch shape: {recon_batch.shape}')
                        raise Exception("Shape of input and reconstructed input does not match!!")
                print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / (len(train_loader.dataset) * epochs)))
                logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / (len(train_loader.dataset) * epochs)))
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            torch.save(model.state_dict(), 'model.pth')
            torch.save(kmeans.state_dict(), 'kmeans.pth')
            # summary_writer.add_scalar('Loss/train_epoch', total_loss / (len(train_loader.dataset) * epochs), epoch)

        torch.save(model.state_dict(), 'model.pth')
        torch.save(kmeans.state_dict(), 'kmeans.pth')
        # Evaluate the trained model
        model.eval()
        total_loss = 0.0
        encoded_vectors = []
        assignments_list = []
        logging.info("Evaluating the model")
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                image, masked_image = data[0].to(device), data[1].to(device)  # Move data to GPU if available
                # data = data[0].to(device)  # Move data to GPU if available
                encoded, recon_batch, mu, logvar = model(image)
                if image.shape == recon_batch.shape:
                    loss, ssim = loss_func.loss_function(recon_batch, image, mu, logvar)
                    total_loss += loss.item()
                    total_ssim += ssim.item()
                    centroids, assignments = kmeans(encoded.detach())
                    encoded_vectors.append(encoded)
                    assignments_list.append(assignments)
                else:
                    print(f'Input batch shape: {data.shape}')
                    print(f'Reconstructed batch shape: {recon_batch.shape}')
                    raise Exception("Shape of input and reconstructed input does not match!!")
            encoded_vectors = torch.cat(encoded_vectors)
            assignments_list = torch.cat(assignments_list)
            plot_tsneV1(encoded_vectors, assignments_list, filename='tsne_plot_evaluate.png')
        average_loss = total_loss / len(test_loader.dataset)
        average_ssim = total_ssim / len(test_loader.dataset)
        # print(f'====> Test set loss: {average_loss:.4f}')
        # print(f'====> Test set SSIM: {average_ssim:.4f}')
        logging.info(f'====> Test set loss: {average_loss:.4f}')
        logging.info(f'====> Test set SSIM: {average_ssim:.4f}')
        return average_loss

    # Function to load and plot images
    def plot_images(paths, cluster_num):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        for idx, path in enumerate(paths):
            img = Image.open(path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Image {idx + 1}")
            axes[idx].axis('off')
        plt.suptitle(f"Cluster {cluster_num} - Top 10 Images")
        plt.savefig(f'cluster_{cluster_num}.png')

    # Function to load and plot images in a grid
    def plot_all_clusters(cluster_assignments, num_clusters):
        fig, axes = plt.subplots(num_clusters, 10, figsize=(20, 2 * num_clusters))
        
        for cluster_num, (key, value) in enumerate(cluster_assignments.items()):
            combined = list(zip(value['distance'], value['paths']))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            top_10_combined = sorted_combined[:10]  # Get top 10 paths and distances
            
            for idx, (distance, path) in enumerate(top_10_combined):
                img = Image.open(path)
                axes[cluster_num, idx].imshow(img)
                axes[cluster_num, idx].set_title(f"{distance:.2f}", fontsize=8)
                axes[cluster_num, idx].axis('off')
            
            # Add cluster number as a super title for the row
            axes[cluster_num, 0].annotate(f"Cluster {key}", xy=(0, 0.5), xytext=(-axes[cluster_num, 0].yaxis.labelpad - 5, 0),
                                        xycoords=axes[cluster_num, 0].yaxis.label, textcoords='offset points',
                                        size='large', ha='right', va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle("Top 10 Images for Each Cluster", y=0.98, fontsize=16)
        plt.savefig('all_clusters.png')

    def highlight_patch_location(original_image_path, patch_image_path):
        # pdb.set_trace()
        if not os.path.exists("Highlighted_Images"):
            os.makedirs("Highlighted_Images")
        # else:
        #     print("Directory already exists")
        #     shutil.rmtree("Highlighted_Images")
        #     os.makedirs("Highlighted_Images")
        # Load the original image and the patch image
        # print(f"Original image path: {original_image_path}")
        original_image = cv2.imread(original_image_path)
        patch_image = cv2.imread(patch_image_path)

        if original_image is None or patch_image is None:
            print("Error: Could not load one of the images.")
            return

        # Convert images to grayscale
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_patch = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray_original, gray_patch, cv2.TM_CCOEFF_NORMED)

        # Get the best match position
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Get the dimensions of the patch image
        patch_height, patch_width = gray_patch.shape

        # Define the rectangle coordinates
        top_left = max_loc
        bottom_right = (top_left[0] + patch_width, top_left[1] + patch_height)

        # Draw a rectangle around the detected patch
        highlighted_image = original_image.copy()
        cv2.rectangle(highlighted_image, top_left, bottom_right, (0, 0, 255), 50)

        # Display the images
        pdb.set_trace()
        class_label = original_image_path.split('/')[-2]
        class_dir = os.path.join("Highlighted_Images", class_label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        highlighted_image_name = f"{class_dir}/" + original_image_path.split('/')[-1]
        print(f"Saving highlighted image to: {highlighted_image_name}")
        cv2.imwrite(highlighted_image_name, highlighted_image)

    def visualize_class_images(class_folders, num_columns=5):
        def load_images_from_folder(folder):
            images = []
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                if os.path.isfile(img_path):
                    images.append(img_path)
            return images

        def create_image_grid(image_paths, num_columns):
            num_images = len(image_paths)
            num_rows = (num_images // num_columns) + int(num_images % num_columns > 0)
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 2 * num_rows))
            axes = axes.ravel()

            for idx, img_path in enumerate(image_paths):
                img = Image.open(img_path)
                axes[idx].imshow(img)
                # Remove title to avoid showing image name
                axes[idx].axis('off')

            for i in range(num_images, num_rows * num_columns):
                axes[i].axis('off')

            plt.tight_layout()
            return fig

        def plot_combined_image_grid(class_folders, num_columns):
            figs = []
            for class_label, folder in class_folders.items():
                image_paths = load_images_from_folder(folder)
                fig = create_image_grid(image_paths, num_columns)
                figs.append((fig, class_label))

            combined_fig = plt.figure(figsize=(20, 2 * sum([f[0].get_size_inches()[1] for f in figs])))

            for i, (fig, class_label) in enumerate(figs):
                ax = combined_fig.add_subplot(len(figs), 1, i + 1)
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Class {class_label} Images", fontsize=25)

            plt.tight_layout()
            plt.savefig('combined_image_grid.png')

        # Execute the plotting function
        plot_combined_image_grid(class_folders, num_columns)

    
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
        test_dataset = ExplanationsPatchesDataset(txt_file="gradcam_patches_CFH.txt", root_dir=".", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=None, shuffle=True)

        model = AutoencoderKMeans(input_dim, latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model.to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        # kmeans = KMeans(num_clusters, latent_dim).to(device)
        # kmeans.load_state_dict(torch.load('kmeans.pth'))
        encoded_vectors = []
        assignments_list = []
        labels = []
        paths = []
        # cluster_assignments = {'0': [], '1': [], '2': []}
        logging.info("Evaluation mode: Generating explanations clusters")
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # pdb.set_trace()
                image = data[0].to(device)
                label = data[1]
                path = data[2]
                encoded, recon_batch, mu, logvar = model(image)
                # centroids, assignments = kmeans(encoded.detach())
                # kmeans.update_centroids(encoded.detach(), assignments)
                encoded_vectors.append(encoded)
                # assignments_list.append(assignments)
                labels.append(label)
                paths.append(path)
            #     for i, assign in enumerate(assignments):
            #         cluster_assignments[str(assign.item())].append(label[i])
            encoded_vectors = torch.cat(encoded_vectors)
            # Fit GMM
            gmm = GaussianMixture(n_components=num_clusters, covariance_type='full')
            gmm.fit(encoded_vectors.cpu().numpy())
            assignments_gmm = gmm.predict(encoded_vectors.cpu().numpy())
            # pdb.set_trace()
            flattened_labels = [string for tuple in labels for string in tuple]
            paths = [string for tuple in paths for string in tuple]
            assert len(assignments_gmm) == len(flattened_labels), "Length of assignments and labels do not match"
            # Initialize dictionary to store cluster assignments
            cluster_assignments = {str(i): {'label':[], 'distance':[], 'paths':[]} for i in range(num_clusters)}
            # Calculate and store distances from the centroid
            for i, assign in enumerate(assignments_gmm):
                cluster_assignments[str(assign)]['label'].append(flattened_labels[i])
                cluster_assignments[str(assign)]['paths'].append(paths[i])
                # Calculate distance from the centroid
                distance = np.linalg.norm(encoded_vectors[i].cpu().numpy() - gmm.means_[assign])
                # print(f'GMM means: {gmm.means_[assign]}')
                cluster_assignments[str(assign)]['distance'].append(distance)

            # Sort paths by distance within each cluster and print top 10
            closest_10_paths = []
            for key, value in cluster_assignments.items():
                # Combine distances and paths and sort them
                combined = list(zip(value['distance'], value['paths']))
                sorted_combined = sorted(combined, key=lambda x: x[0])
                top_10_combined = sorted_combined[:10]  # Get top 10 paths and distances
                
                # Print top 10 paths and distances for each cluster
                print(f'Cluster {key}:')
                for distance, path in top_10_combined:
                    print(f'Path: {path}, Distance: {distance}')
                
                # Extract the top 10 paths for plotting
                top_10_paths = [path for _, path in top_10_combined]
                closest_10_paths.append(top_10_paths)

                # pdb.set_trace()
                
                # Plot the top 10 images for the current cluster
                # plot_images(top_10_paths, key)
            
            # Plot all clusters
            plot_all_clusters(cluster_assignments, num_clusters)
            
            # Print cluster information
            with open("class-cluster-CFH.txt", "w") as f:
                for key, value in cluster_assignments.items():
                    counts = {}
                    for item in value['label']:
                        if item == '0':
                            label = 'Class 0'
                            counts[label] = counts.get(label, 0) + 1
                        elif item == '1':
                            label = 'Class 1'
                            counts[label] = counts.get(label, 0) + 1
                        elif item == '2':
                            label = 'Class 2'
                            counts[label] = counts.get(label, 0) + 1
                    print(f'Cluster {key}: {counts}')
                    f.write(f'Cluster {key}: {counts}\n')
            # plot the t-SNE plot here
            plot_tsneV1(encoded_vectors, torch.tensor(assignments_gmm), filename='tsne_plot_CFH.png')

            # Highlight the patch location in the original image
            # pdb.set_trace()
            images_processed = []  # Using a set for better performance on membership checks
            print("Processing explanations: ", len(paths))
            # pdb.set_trace()
            all_paths = [path for sublist in closest_10_paths for path in sublist]
            print("Number of explanations: ", len(all_paths))
            # pdb.set_trace()
            # Tracing explanations for each original image
            for path in all_paths:
                class_label = path.split('/')[2]
                explainability_folder = path.split('/')[1]
                
                # Constructing the image path based on folder type
                if explainability_folder.startswith("gradcam_patches") or explainability_folder.startswith("scorecam_patches"):
                    image_name = path.split('/')[3] + ".jpg"
                    image_path = os.path.join('..', 'deepcover', 'data', 'Fundus_correct_CFH', class_label, image_name)
                elif explainability_folder == 'patches':
                    image_name = "TE-" + path.split('/')[3] + ".jpg"
                    image_path = os.path.join('..', 'deepcover', 'data', 'Fundus', class_label, image_name)
                else:
                    continue  # Skip unknown folders
                print(f'Processing image: {image_path}')
                highlight_patch_location(image_path, path)
                images_processed.append(image_path)
                # Process image if it has not been processed yet
                # if image_path not in images_processed:
                #     print(f'Processing image: {image_path}')
                #     highlight_patch_location(image_path, path)
                #     images_processed.add(image_path)
            
            print("Number of explanations processed: ", len(images_processed))
            visualize_class_images({'0': 'Highlighted_Images/0', '1': 'Highlighted_Images/1', '2': 'Highlighted_Images/2'}, 10)
    
    if not os.path.exists('model.pth') and not os.path.exists('kmeans.pth'):
        train_vae({"batch_size": 128, "latent_dim": 350})
    else:
        infer_vae({"batch_size": 128, "latent_dim": 350})

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
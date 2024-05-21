import torch
import torch.nn as nn
import pdb

class KMeans(nn.Module):
    def __init__(self, num_clusters, input_dim):
        super(KMeans, self).__init__()
        self.num_clusters = num_clusters
        self.centroids = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(num_clusters, input_dim)))
        # self.centroids = nn.Parameter(torch.rand(num_clusters, input_dim))

    def update_centroids(self, x, assignments):
        # Create an empty tensor for the new centroids
        new_centroids = torch.empty(self.num_clusters, x.size(1), device=x.device)

        # Compute the new centroid for each cluster
        for i in range(self.num_clusters):
            # Find all points assigned to cluster i
            assigned_points = x[assignments == i]
            
            # Check if any points were assigned to this cluster
            if assigned_points.size(0) > 0:
                # Compute the mean of the assigned points
                new_centroids[i] = assigned_points.mean(dim=0)
            else:
                # If no points were assigned to this cluster, leave the centroid as is
                new_centroids[i] = self.centroids[i]

        # Update the centroids
        self.centroids.data = new_centroids
    
    def forward(self, x):
        # Reshape input for broadcasting
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centroids = self.centroids.unsqueeze(0)  # [1, num_clusters, input_dim]
                
        # Compute distances
        distances = torch.norm(x - centroids, dim=2)  # [batch_size, num_clusters]
        
        # Assign each point to the nearest centroid
        _, assignments = torch.min(distances, dim=1)  # [batch_size]
        self.update_centroids(x.squeeze(1), assignments)
        
        return centroids, assignments


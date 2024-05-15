import torch
import torch.nn as nn

class KMeans(nn.Module):
    def __init__(self, num_clusters, input_dim):
        super(KMeans, self).__init__()
        self.num_clusters = num_clusters
        self.centroids = nn.Parameter(torch.rand(num_clusters, input_dim))
    
    def forward(self, x):
        # Reshape input for broadcasting
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centroids = self.centroids.unsqueeze(0)  # [1, num_clusters, input_dim]
        
        # Compute distances
        distances = torch.norm(x - centroids, dim=2)  # [batch_size, num_clusters]
        
        # Assign each point to the nearest centroid
        _, assignments = torch.min(distances, dim=1)  # [batch_size]
        
        return centroids, assignments

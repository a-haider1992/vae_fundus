import torch
import torch.nn as nn

class KMeans(nn.Module):
    def __init__(self, num_clusters, input_dim):
        super(KMeans, self).__init__()
        self.num_clusters = num_clusters
        self.centroids = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(num_clusters, input_dim)))
        self.register_buffer('centroid_sums', torch.zeros(num_clusters, input_dim))
        self.register_buffer('counts', torch.zeros(num_clusters))

    def update_centroids(self, x, assignments):
        batch_centroid_sums = torch.zeros_like(self.centroid_sums)
        batch_counts = torch.zeros_like(self.counts)
        
        for i in range(self.num_clusters):
            assigned_points = x[assignments == i]
            if assigned_points.size(0) > 0:
                batch_centroid_sums[i] = assigned_points.sum(dim=0)
                batch_counts[i] = assigned_points.size(0)
        
        self.centroid_sums += batch_centroid_sums
        self.counts += batch_counts
        
        # Update centroids based on cumulative sums and counts
        mask = self.counts > 0
        self.centroids.data[mask] = self.centroid_sums[mask] / self.counts[mask].unsqueeze(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        centroids = self.centroids.unsqueeze(0)
        distances = torch.norm(x - centroids, dim=2)
        _, assignments = torch.min(distances, dim=1)
        return centroids, assignments

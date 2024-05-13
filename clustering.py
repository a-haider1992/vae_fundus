import numpy as np

def k_means_clustering(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = np.random.choice(data, size=k)
    
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.abs(data[:, np.newaxis] - centroids), axis=1)
        
        # Update centroids based on the mean of the assigned data points
        new_centroids = np.array([np.mean(data[labels == i]) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids
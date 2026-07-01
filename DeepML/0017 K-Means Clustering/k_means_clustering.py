import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)

    for _ in range(max_iterations):
        grouping = calculate_grouping(points_t, k, centroids)
        new_centroid = calculate_centroid(grouping)
        if torch.allclose( centroids, new_centroid, atol=1e-4):
            break
        centroids = new_centroid
        
    return centroids

def calculate_grouping(points_t, k, centroids):
    grouping = [[] for i in range(k)]
    for point_t in points_t:
        # calculate distance from all centroid
        distances = [torch.dist(centroids[i], point_t) for i in range(k)]
        closest_centroid = torch.argmin(torch.tensor(distances)).item()

        grouping[closest_centroid].append(point_t)

    return grouping

def calculate_centroid(grouping):

    new_centroids = []

    for group in grouping:
        group_t = torch.stack(group)
        centroid = torch.mean(group_t, dim = 0)
        new_centroids.append(centroid)

    return torch.stack(new_centroids)
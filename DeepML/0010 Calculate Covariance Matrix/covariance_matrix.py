import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)

    v_mean = torch.mean(v_t, dim=1)

    # and a dimension at position 1 so that we can do broadcasting
    v_centered = v_t - v_mean.unsqueeze(1)

    return (v_centered @ v_centered.T) / len((v_t) - 1)
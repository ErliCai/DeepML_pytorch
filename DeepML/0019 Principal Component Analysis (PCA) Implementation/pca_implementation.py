import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """

    data_t = torch.tensor(data)
    mean, std = data_t.mean(dim=0), data_t.std(dim=0)

    standardized = (data_t - mean)/ std
    covariance = torch.cov(standardized.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    top_values, top_indices = torch.topk(eigenvalues, k)

    result = torch.stack([eigenvectors[top_indices[i]] for i in range(k)])
    result = result.T

    for i in range(result.shape[1]):
        component = result[:, i]
        nonzero = component[component != 0]
        if nonzero.numel() > 0 and nonzero[0].item() < 0:
            result[:, i] = -component

    print(result)
    return result

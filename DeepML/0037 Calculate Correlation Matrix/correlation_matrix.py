import torch
from numpy import ndarray
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, ndarray],
    Y: Optional[Union[torch.Tensor, list, ndarray]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """

    X_t = torch.tensor(X,dtype=torch.float)
    if Y is None:
        cov = torch.cov(X_t.T)
        std = torch.std(X_t, dim=0)
        return cov / torch.outer(std, std)
    else:
        Y_t = torch.tensor(Y, dtype=torch.float)

        data = torch.cat([X_t, Y_t], dim=0)

        cov = torch.cov(data.T)
        std_X = torch.std(X_t)
        std_Y = torch.std(Y_t)

        return cov/ std_X / std_Y
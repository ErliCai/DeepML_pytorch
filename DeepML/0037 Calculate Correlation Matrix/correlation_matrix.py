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
    # Your implementation here
    pass

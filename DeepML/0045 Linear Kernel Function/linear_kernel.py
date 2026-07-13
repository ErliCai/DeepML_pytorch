import torch
from numpy import ndarray
from typing import Union

def linear_kernel(x1: Union[torch.Tensor, list, ndarray],
                  x2: Union[torch.Tensor, list, ndarray]) -> float:
    """
    Compute the linear kernel between two vectors.

    Args:
        x1: First input vector.
        x2: Second input vector.

    Returns:
        Scalar dot product of x1 and x2.
    """

    # TODO
    pass

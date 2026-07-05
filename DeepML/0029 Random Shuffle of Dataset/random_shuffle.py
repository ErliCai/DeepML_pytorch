import torch
from typing import Tuple
import numpy as np

def shuffle_data(X, y, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly shuffle X and y together, maintaining the correspondence between samples.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X_s, y_s = X[idx], y[idx]

    return torch.tensor(X_s), torch.tensor(y_s)
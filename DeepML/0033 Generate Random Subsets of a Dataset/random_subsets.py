import numpy as np
from typing import List, Tuple

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """

    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)
    size = n if replacements else n//2
    out = []
    for _ in range(n_subsets):
        idx = np.random.choice(n, size, replacements)
        out.append((X[idx].tolist(), y[idx].tolist()))
    return out
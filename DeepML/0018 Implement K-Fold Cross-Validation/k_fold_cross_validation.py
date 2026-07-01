import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """

    samples = [i for i in range(n_samples)]
    if shuffle:
        samples = np.random.shuffle(samples)

    d, r = n_samples//k, n_samples%k
    splits = []

    init_index = 0
    for i in range(k):
        next_index = (i+1) * d + min((i+1),r)
        splits.append(samples[init_index:next_index])
        init_index = next_index

    k_fold = []
    for i in range(k):
        training = [splits[j] for j in range(k) if j != i]
        training = np.array(training).flatten().tolist()
        validation = splits[i]
        k_fold.append((training, validation))

    return k_fold

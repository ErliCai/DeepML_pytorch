import torch
import math
from typing import List, Dict

def adaboost_fit(X, y, n_clf) -> List[Dict]:
    """
    Fit an AdaBoost classifier using PyTorch tensors (no sklearn).
    Args:
        X: torch.Tensor or array-like, shape (n_samples, n_features)
        y: torch.Tensor or array-like, shape (n_samples,) with labels (+1/-1)
        n_clf: int, number of weak classifiers
    Returns:
        List of dictionaries with classifier params: 'polarity', 'threshold', 'feature_index', 'alpha'.
    """

    # initialize classifier
    classifier = [{'polarity':1, 'threshold':0, 'feature_index':0, 'alpha':1/n_clf} for i in range(n_clf)]
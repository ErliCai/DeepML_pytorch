import torch
from typing import List

def divide_on_feature(X, feature_i, threshold) -> List[torch.Tensor]:
    """
    Divide the tensor X into two subsets based on whether X[:, feature_i] is greater than or equal to (or equal to) the threshold.
    Return two tensors: one with samples that meet the condition, one with samples that do not.
    """

    X = torch.tensor(X)
    X_satisfy, X_not = [], []
    for x in X:
        if x[feature_i] >= threshold:
            X_satisfy.append(x)
        else:
            X_not.append(x)

    X_satisfy = torch.stack(X_satisfy)
    X_not = torch.stack(X_not)

    return X_satisfy, X_not

def divide_on_feature2(X, feature_i, threshold) -> List[torch.Tensor]:
    """
    Divide the tensor X into two subsets based on whether X[:, feature_i] is greater than or equal to (or equal to) the threshold.
    Return two tensors: one with samples that meet the condition, one with samples that do not.
    """

    X_t = torch.tensor(X)
    mask = X_t[:,feature_i] >= threshold

    subset1 = X_t[mask]
    subset2 = X_t[~mask]

    return subset1, subset2
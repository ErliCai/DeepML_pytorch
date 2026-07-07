import torch
from numpy import ndarray
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, ndarray],
                   y_pred: Union[torch.Tensor, list, ndarray]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    correct = y_true == y_pred
    print(correct)
    return correct.sum().item() / len(correct)
import torch

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the precision metric for binary classification.
    
    Precision is defined as the ratio of true positives to the sum of 
    true positives and false positives.
    
    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)
    
    Returns:
        Precision value as a scalar tensor
    """

    positive = y_true == 1
    negative = y_true == 0
    True_positive = y_pred[positive].sum()
    False_positive = y_pred[negative].sum()

    if True_positive == 0 and False_positive == 0:
        return 0

    return True_positive / (True_positive + False_positive)
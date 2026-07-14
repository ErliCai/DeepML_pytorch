import torch

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_pred: Tensor of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """

    positive = y_pred == 1
    negative = y_pred == 0

    TP = y_true[positive].sum()
    FN = y_true[negative].sum()

    recall = TP / (TP + FN)

    return float(recall)
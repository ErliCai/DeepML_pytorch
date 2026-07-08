import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """

    scores_t = torch.tensor(scores)

    score_max = scores_t.max() 
    logSum = torch.log(torch.exp(scores_t- score_max).sum())
    log_softmax = scores_t - score_max - logSum

    return log_softmax
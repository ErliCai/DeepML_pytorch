import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """

    scores_exps = [torch.exp(torch.tensor(score)) for score in scores]
    scores_exps = torch.stack(scores_exps)
    scores_exps_sum = sum(scores_exps)
    scores_exps /= scores_exps_sum

    return scores_exps.tolist()
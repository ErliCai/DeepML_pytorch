import torch


def mean_ablate(activations: torch.Tensor, mask: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """
    Apply mean ablation to node activations using PyTorch.

    Args:
        activations: Original node activations, shape (n,) or (batch, n)
        mask: Binary mask where 1 = ablate (replace with mean), 0 = keep original
        means: Precomputed mean activations for each node, shape (n,)

    Returns:
        Ablated activations with same shape as input
    """
    
    return mask * means + (1-mask) * activations



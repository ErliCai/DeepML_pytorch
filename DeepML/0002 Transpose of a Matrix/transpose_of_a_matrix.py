import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)

    a_transpose = torch.empty(a_t.size(1), a_t.size(0))

    for i in range(a_t.size(0)):
        for j in range(a_t.size(1)):
            a_transpose[j,i] = a_t[i,j]

    return a_transpose
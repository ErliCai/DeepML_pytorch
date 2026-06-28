import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    
    a = torch.tensor(a)
    b = torch.tensor(b)

    if len(a[0]) != len(b):
        return torch.tensor(-1)
    
    result = torch.empty(len(a),len(b[0]))

    for i in range(len(a)):
        for j in range(len(b[0])):
            result[i,j] = sum([a[i,k]*b[k,j] for k in range(len(a[0]))])

    return result
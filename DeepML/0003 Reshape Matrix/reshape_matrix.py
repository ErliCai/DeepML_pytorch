import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor on mismatch.
    """
    # Dimension check
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])
    # Convert to tensor and reshape
    a_t = torch.as_tensor(a, dtype=torch.float)
      
    result = torch.empty(new_shape)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            pos = i * new_shape[1] + j
            original_row, original_column = pos // len(a[1]),  pos % len(a[1])
            result[i][j] = a_t[original_row, original_column]

    ## result[0,0] = a_t[0, 0], result[0,1] = a_t[0, 1]
    return result

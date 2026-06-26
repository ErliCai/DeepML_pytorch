import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """

    matrix = torch.tensor(matrix)

    trace = matrix[0, 0] + matrix[1, 1]
    det = matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1]

    lambda1 = (trace + torch.sqrt(trace**2 - 4*det)) / 2
    lambda2 = (trace - torch.sqrt(trace**2 - 4*det)) / 2

    return torch.tensor([lambda1, lambda2])
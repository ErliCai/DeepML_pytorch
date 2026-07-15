import torch

def rref(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a matrix to its Reduced Row Echelon Form (RREF).
    
    RREF is achieved through elementary row operations:
    - Each leading entry in a row is 1 (pivot)
    - All other entries in a pivot column are 0
    - Leading 1s move to the right as you go down rows
    
    Args:
        matrix: Input matrix as a 2D tensor
    
    Returns:
        Matrix in RREF form
    """

    m, n = matrix.shape
    no_pivot = 0
    for i in range(n):
        if i >= m:
            break

        found_pivot = False

        for j in range(i,m):
            if matrix[j, i] != 0:
                found_pivot = True
                matrix[[i - no_pivot, j]] = matrix[[j, i - no_pivot]].clone()
                break
        if not found_pivot:
            no_pivot += 1
            continue

        matrix[i-no_pivot] = matrix[i-no_pivot] / matrix[i-no_pivot,i]

        for j in range(m):
            if j != i-no_pivot:
                matrix[j] -= matrix[i-no_pivot] * matrix[j,i]

    return matrix


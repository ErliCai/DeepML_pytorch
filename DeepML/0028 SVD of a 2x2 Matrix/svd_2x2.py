import torch

def svd_2x2(A: list[list[float]]) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """
    Compute SVD of a 2x2 matrix without using built-in SVD functions.
    Returns U, s, V such that A = U @ diag(s) @ V.

    Args:
        A: 2x2 matrix as nested list

    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: singular values in descending order
        V: 2x2 matrix (rows are right singular vectors)
    """

    A_t = torch.as_tensor(A, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(A_t.T @ A_t)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    S = eigenvalues**(1/2)

    V = eigenvectors
    U = A_t @ V / S

    return U.float(), S.float(), V.T.float()
import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """

    eigenvalues , eigenvectors = torch.linalg.eig(A.T @ A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    eigenvalues, indices = torch.sort(eigenvalues, descending=True)
    S = eigenvalues ** (1/2)

    Vt = torch.stack([eigenvectors[:,indices[0]], eigenvectors[:,indices[1]]])
    U =  A @ Vt.T @ torch.diag(1/S)

    S = S.real
    U = U.real
    Vt = Vt.real

    print(U @ S @ Vt)

    return U, S, Vt


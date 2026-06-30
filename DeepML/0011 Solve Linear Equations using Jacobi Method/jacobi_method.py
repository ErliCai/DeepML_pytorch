import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    jacobi = torch.zeros_like(A_t)
    D = [b_t[i]/A_t[i][i] for i in range(len(A_t))]
    D = torch.tensor(D)

    for i in range(len(A_t)):
        for j in range(len(A_t)):
            if j != i:
                jacobi[i][j] = - A_t[i][j] / A_t[i][i]

    result = torch.zeros(len(A_t))
    for i in range(n):
        result = jacobi @ result + D

    return result.round(decimals=4)

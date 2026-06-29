import torch

from jacobi_method import solve_jacobi


def test_jacobi_method():
    A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
    b = [-1, 2, 3]
    n = 2

    result = solve_jacobi(A, b, n)

    assert torch.allclose(result, torch.tensor([0.146, 0.2032, -0.5175]), atol=1e-4)


if __name__ == "__main__":
    test_jacobi_method()
    print("All tests passed")

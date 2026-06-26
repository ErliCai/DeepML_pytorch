import torch

from calculate_eigenvalues_of_a_matrix import calculate_eigenvalues


def test_calculate_eigenvalues():
    matrix = [[2, 1], [1, 2]]

    result = calculate_eigenvalues(matrix)

    assert torch.allclose(result, torch.tensor([3.0, 1.0]))


if __name__ == "__main__":
    test_calculate_eigenvalues()
    print("All tests passed")
import torch

from matrix_transformation import transform_matrix


def test_matrix_transformation():
    A = [[1, 2], [3, 4]]
    T = [[2, 0], [0, 2]]
    S = [[1, 1], [0, 1]]

    result = transform_matrix(A, T, S)

    assert torch.equal(result, torch.tensor([[0.5, 1.5], [1.5, 3.5]]))


if __name__ == "__main__":
    test_matrix_transformation()
    print("All tests passed")

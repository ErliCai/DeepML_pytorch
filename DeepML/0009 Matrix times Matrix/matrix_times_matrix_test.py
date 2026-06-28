import torch

from matrix_times_matrix import matrixmul


def test_matrix_times_matrix():
    A = [[1, 2], [2, 4]]
    B = [[2, 1], [3, 4]]

    result = matrixmul(A, B)

    assert torch.equal(result, torch.tensor([[8, 9], [16, 18]]))


if __name__ == "__main__":
    test_matrix_times_matrix()
    print("All tests passed")

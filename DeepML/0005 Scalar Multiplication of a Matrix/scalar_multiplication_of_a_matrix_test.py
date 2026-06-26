import torch

from scalar_multiplication_of_a_matrix import scalar_multiply


def test_scalar_multiply():
    matrix = [[1, 2], [3, 4]]
    scalar = 2

    result = scalar_multiply(matrix, scalar)

    assert torch.equal(result, torch.tensor([[2., 4.], [6., 8.]]))


if __name__ == "__main__":
    test_scalar_multiply()
    print("All tests passed")
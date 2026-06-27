import torch

from matrix_inverse import inverse_2x2


def test_matrix_inverse():
    matrix = [[4, 7], [2, 6]]

    result = inverse_2x2(matrix)

    print(result)

    assert torch.allclose(result, torch.tensor([[0.6, -0.7], [-0.2, 0.4]]), atol=1e-4)


if __name__ == "__main__":
    test_matrix_inverse()
    print("All tests passed")

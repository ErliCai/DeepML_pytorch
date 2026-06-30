import torch

from determinant_4x4 import determinant_4x4


def test_determinant_4x4():
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    result = determinant_4x4(a)

    assert result == 0


def test_determinant_4x4_diagonal():
    a = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0]))

    result = determinant_4x4(a)

    assert result == 120


if __name__ == "__main__":
    test_determinant_4x4()
    test_determinant_4x4_diagonal()
    print("All tests passed")

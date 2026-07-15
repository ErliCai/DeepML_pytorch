import torch

from rref import rref


def test_rref():
    matrix = torch.tensor([
        [1, 2, -1, -4],
        [2, 3, -1, -11],
        [-2, 0, -3, 22],
    ], dtype=torch.float)
    result = rref(matrix)
    print(result)

    expected = torch.tensor([
        [1.0, 0.0, 0.0, -8.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, -2.0],
    ])
    assert torch.allclose(torch.as_tensor(result), expected)


def test_rref_pivot_swap():
    matrix = torch.tensor([
        [0.0, 2.0, -1.0, -4.0],
        [2.0, 0.0, -1.0, -11.0],
        [-2.0, 0.0, 0.0, 22.0],
    ])
    result = rref(matrix)
    print(result.numpy().tolist())

    expected = torch.tensor([
        [1.0, 0.0, 0.0, -11.0],
        [0.0, 1.0, 0.0, -7.5],
        [0.0, 0.0, 1.0, -11.0],
    ])
    assert torch.allclose(torch.as_tensor(result), expected)


def test_rref_rank_deficient():
    matrix = torch.tensor([
        [1.0, 2.0, -1.0],
        [2.0, 4.0, -1.0],
        [-2.0, -4.0, -3.0],
    ])
    result = rref(matrix)
    print(result.numpy().tolist())

    expected = torch.tensor([
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    assert torch.allclose(torch.as_tensor(result), expected)


if __name__ == "__main__":
    test_rref()
    test_rref_pivot_swap()
    test_rref_rank_deficient()
    print("All tests passed")

import torch

from conv2d import simple_conv2d


def test_simple_conv2d():
    input_matrix = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.float)
    kernel = torch.tensor([
        [1, 0],
        [-1, 1],
    ], dtype=torch.float)
    padding = 1
    stride = 2

    output = simple_conv2d(input_matrix, kernel, padding, stride)
    print(output)

    expected = torch.tensor([
        [1.0, 1.0, -4.0],
        [9.0, 7.0, -4.0],
        [0.0, 14.0, 16.0],
    ])
    assert torch.allclose(output.squeeze(), expected)


if __name__ == "__main__":
    test_simple_conv2d()
    print("All tests passed")

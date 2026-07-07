import torch

from one_hot_encoding import to_categorical


def test_one_hot_encode():
    x = torch.tensor([0, 1, 2, 1, 0])
    output = to_categorical(x)

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.float,
    )

    assert torch.equal(output, expected)


if __name__ == "__main__":
    test_one_hot_encode()
    print("All tests passed")
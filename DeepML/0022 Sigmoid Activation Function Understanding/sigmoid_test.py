import torch

from sigmoid import sigmoid


def test_sigmoid():
    result = sigmoid(0)

    assert result == 0.5


if __name__ == "__main__":
    test_sigmoid()
    print("All tests passed")

import torch

from divide_dataset import divide_on_feature


def test_divide_dataset():
    X = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ])
    feature_i = 0
    threshold = 5

    result = divide_on_feature(X, feature_i, threshold)

    assert len(result) == 2
    assert torch.equal(result[0], torch.tensor([[5, 6], [7, 8], [9, 10]]))
    assert torch.equal(result[1], torch.tensor([[1, 2], [3, 4]]))


if __name__ == "__main__":
    test_divide_dataset()
    print("All tests passed")
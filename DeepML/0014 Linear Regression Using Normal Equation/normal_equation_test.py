import torch

from normal_equation import linear_regression_normal_equation


def test_normal_equation():
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]

    result = linear_regression_normal_equation(X, y)

    print(result)

    assert torch.allclose(result, torch.tensor([0.0, 1.0]), atol=1e-4)


if __name__ == "__main__":
    test_normal_equation()
    print("All tests passed")

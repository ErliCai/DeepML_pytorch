import torch

from linear_regression_gradient_descent import linear_regression_gradient_descent


def test_linear_regression_gradient_descent():
    X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = torch.tensor([3.0, 5.0, 7.0])

    result = linear_regression_gradient_descent(X, y, alpha=0.1, iterations=1000)

    assert torch.allclose(result, torch.tensor([1.0, 2.0]), atol=1e-4)


if __name__ == "__main__":
    test_linear_regression_gradient_descent()
    print("All tests passed")
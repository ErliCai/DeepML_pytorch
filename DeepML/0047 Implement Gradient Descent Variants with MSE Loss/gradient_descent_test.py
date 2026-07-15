import torch

from gradient_descent import gradient_descent


def test_gradient_descent():
    X = torch.tensor([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=torch.float)
    y = torch.tensor([2, 3, 4, 5], dtype=torch.float)

    learning_rate = 0.01
    n_epochs = 1000
    batch_size = 2

    for method, kwargs in [
        ("batch", {}),
        ("stochastic", {}),
        ("mini_batch", {"batch_size": batch_size}),
    ]:
        weights = torch.zeros(X.shape[1])
        result = gradient_descent(
            X, y, weights, learning_rate, n_epochs, method=method, **kwargs
        )
        result = torch.as_tensor(result)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()


if __name__ == "__main__":
    test_gradient_descent()
    print("All tests passed")

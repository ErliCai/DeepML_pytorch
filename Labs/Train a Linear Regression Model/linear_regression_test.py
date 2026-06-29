import numpy as np

from LinearRegressionModel import train


def test_mse_improves():
    np.random.seed(42)
    X = np.random.randn(200, 3)
    true_w = np.array([1.0, -2.0, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(200)
    W = np.random.randn(3) * 0.01
    b = 0.0

    W_out, b_out = train(X, y, W, b)

    mse_before = np.mean((y - (X @ W + b)) ** 2)
    mse_after = np.mean((y - (X @ W_out + b_out)) ** 2)

    print(mse_after, mse_before)

    assert mse_after < mse_before * 0.1


if __name__ == "__main__":
    test_mse_improves()
    print("All tests passed")

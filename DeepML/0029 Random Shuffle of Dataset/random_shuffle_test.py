import numpy as np
from random_shuffle import shuffle_data


def test_example():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    X_s, y_s = shuffle_data(X, y, seed=42)
    assert X_s.shape == X.shape
    assert y_s.shape == y.shape
    for i in range(len(y_s)):
        assert np.array_equal(X_s[i], X[y_s[i] - 1])


if __name__ == "__main__":
    test_example()
    print("All tests passed")

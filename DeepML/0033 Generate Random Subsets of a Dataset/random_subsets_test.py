import numpy as np
from random_subsets import get_random_subsets


def test_example():
    np.random.seed(42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    result = get_random_subsets(X, y, n_subsets=3, replacements=False)
    expected = [
        ([[3, 4], [9, 10]], [2, 5]),
        ([[7, 8], [3, 4]], [4, 2]),
        ([[3, 4], [1, 2]], [2, 1]),
    ]
    assert result == expected


if __name__ == "__main__":
    test_example()
    print("All tests passed")

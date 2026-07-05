import numpy as np
from batch_iterator import batch_iterator


def test_example():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    batches = batch_iterator(X, y, batch_size=2)
    expected = [
        [[[1, 2], [3, 4]], [1, 2]],
        [[[5, 6], [7, 8]], [3, 4]],
        [[[9, 10]], [5]],
    ]
    assert len(batches) == len(expected)
    for (X_b, y_b), (X_e, y_e) in zip(batches, expected):
        assert np.array_equal(X_b, X_e)
        assert np.array_equal(y_b, y_e)


if __name__ == "__main__":
    test_example()
    print("All tests passed")

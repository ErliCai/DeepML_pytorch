import numpy as np

from precision import precision


def test_precision():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    assert precision(y_true, y_pred) == 1.0


if __name__ == "__main__":
    test_precision()
    print("All tests passed")

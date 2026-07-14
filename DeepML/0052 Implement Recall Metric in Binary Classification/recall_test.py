import numpy as np

from recall import recall


def test_recall():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    assert recall(y_true, y_pred) == 0.75


if __name__ == "__main__":
    test_recall()
    print("All tests passed")

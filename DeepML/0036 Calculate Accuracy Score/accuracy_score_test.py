import numpy as np

from accuracy_score import accuracy_score


def test_accuracy_score():
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)

    print(output)

    assert output == 0.8333333333333334


if __name__ == "__main__":
    test_accuracy_score()
    print("All tests passed")
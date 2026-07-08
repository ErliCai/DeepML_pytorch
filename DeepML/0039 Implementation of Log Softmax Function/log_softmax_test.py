import numpy as np

from log_softmax import log_softmax


def test_log_softmax():
    A = np.array([1, 2, 3])
    output = log_softmax(A)

    expected = np.array([-2.4076, -1.4076, -0.4076])

    assert np.allclose(output, expected, atol=1e-4)


if __name__ == "__main__":
    test_log_softmax()
    print("All tests passed")
import numpy as np

from correlation_matrix import calculate_correlation_matrix


def test_calculate_correlation_matrix():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    output = calculate_correlation_matrix(X)
    print(output)
    expected = np.array([[1., 1.],
                         [1., 1.]])
    assert np.allclose(output, expected)


if __name__ == "__main__":
    test_calculate_correlation_matrix()
    print("All tests passed")

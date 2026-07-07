import numpy as np
import torch
from polynomial_features import polynomial_features


def test_example():
    X = np.array([[2, 3], [3, 4], [5, 6]])
    result = polynomial_features(X, degree=2)
    expected = torch.tensor([[1., 2., 3., 4., 6., 9.],
                              [1., 3., 4., 9., 12., 16.],
                              [1., 5., 6., 25., 30., 36.]])
    assert torch.equal(result, expected)


if __name__ == "__main__":
    test_example()
    print("All tests passed")

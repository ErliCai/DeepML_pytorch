import numpy as np

from linear_kernel import kernel_function


def test_linear_kernel():
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    result = kernel_function(x1, x2)
    print(result)
    assert result == 32


if __name__ == "__main__":
    test_linear_kernel()
    print("All tests passed")

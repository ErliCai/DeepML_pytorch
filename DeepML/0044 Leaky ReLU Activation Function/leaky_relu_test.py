import numpy as np

from leaky_relu import leaky_relu


def test_leaky_relu():
    assert leaky_relu(0) == 0
    assert leaky_relu(1) == 1
    assert abs(leaky_relu(-1) - (-0.01)) < 1e-6
    assert abs(leaky_relu(-2, alpha=0.1) - (-0.2)) < 1e-6


if __name__ == "__main__":
    test_leaky_relu()
    print("All tests passed")

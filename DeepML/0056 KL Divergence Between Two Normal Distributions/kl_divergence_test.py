import numpy as np

from kl_divergence import kl_divergence_normal


def test_kl_divergence_basic():
    result = kl_divergence_normal(0.0, 1.0, 1.0, 1.0)
    print(result)
    assert np.isclose(result, 0.5, atol=1e-4)


def test_kl_divergence_identical():
    result = kl_divergence_normal(2.0, 1.5, 2.0, 1.5)
    print(result)
    assert np.isclose(result, 0.0, atol=1e-4)


if __name__ == "__main__":
    test_kl_divergence_basic()
    test_kl_divergence_identical()
    print("All tests passed")

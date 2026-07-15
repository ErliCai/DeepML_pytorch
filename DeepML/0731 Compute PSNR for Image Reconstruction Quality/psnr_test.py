import numpy as np

from psnr import compute_psnr


def test_compute_psnr():
    orig = np.array([[0, 0], [0, 0]], dtype=float)
    recon = np.array([[1, 1], [1, 1]], dtype=float)
    result = compute_psnr(orig, recon, 255)
    print(result)
    assert np.isclose(result, 48.1308, atol=1e-4)


def test_compute_psnr_identical():
    orig = np.array([[10, 20], [30, 40]], dtype=float)
    recon = np.array([[10, 20], [30, 40]], dtype=float)
    assert compute_psnr(orig, recon, 255) == float("inf")


def test_compute_psnr_general():
    orig = np.array([[50, 80], [120, 200]], dtype=float)
    recon = np.array([[52, 78], [118, 205]], dtype=float)
    result = round(compute_psnr(orig, recon, 255), 4)
    print(result)
    assert result == 38.4694


if __name__ == "__main__":
    test_compute_psnr()
    test_compute_psnr_identical()
    test_compute_psnr_general()
    print("All tests passed")

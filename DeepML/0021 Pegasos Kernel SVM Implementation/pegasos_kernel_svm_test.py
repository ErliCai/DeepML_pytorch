import torch

from pegasos_kernel_svm import pegasos_kernel_svm, linear_kernel, rbf_kernel


def test_linear_kernel():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 3.0])

    result = linear_kernel(x, y, sigma=None)

    assert torch.allclose(result, torch.tensor(8.0), atol=1e-4)  # 1*2 + 2*3 = 8


def test_rbf_kernel():
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([1.0, 0.0])

    result = rbf_kernel(x, y, sigma=1.0)

    assert torch.allclose(result, torch.tensor(1.0), atol=1e-4)  # identical points -> exp(0) = 1


def test_pegasos_kernel_svm():
    data = torch.tensor([[1, 2], [2, 3], [3, 1], [4, 1]], dtype=torch.float64)
    labels = torch.tensor([1, 1, -1, -1], dtype=torch.float64)

    alphas, bias = pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100)

    assert torch.allclose(torch.tensor(alphas), torch.tensor([100.0, 0.0, -100.0, -100.0], dtype=torch.float64), atol=1e-4)
    assert abs(bias - (-937.4755)) < 1e-4


if __name__ == "__main__":
    test_linear_kernel()
    test_rbf_kernel()
    test_pegasos_kernel_svm()

    print("All tests passed")

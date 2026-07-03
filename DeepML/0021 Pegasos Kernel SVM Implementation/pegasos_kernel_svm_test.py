import torch

from pegasos_kernel_svm import pegasos_kernel_svm


def test_pegasos_kernel_svm():
    data = torch.tensor([[1, 2], [2, 3], [3, 1], [4, 1]], dtype=torch.float)
    labels = torch.tensor([1, 1, -1, -1], dtype=torch.float)

    alphas, bias = pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100)

    assert torch.allclose(torch.tensor(alphas), torch.tensor([2.0, 2.0, 6.0, 1.0]), atol=1e-4)
    assert abs(bias - 36.2027) < 1e-4


if __name__ == "__main__":
    test_pegasos_kernel_svm()
    print("All tests passed")

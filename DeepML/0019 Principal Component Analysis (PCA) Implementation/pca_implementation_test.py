import torch

from pca_implementation import pca


def test_pca():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    k = 1

    result = pca(data, k)

    assert torch.allclose(result, torch.tensor([[0.7071], [0.7071]]), atol=1e-4)


def test_pca_second_example():
    res = pca([[1.0, 6.0], [2.0, 4.0], [3.0, 2.0]], 1)

    assert [[round(val, 4) for val in row] for row in res.tolist()] == [[0.7071], [-0.7071]]


if __name__ == "__main__":
    test_pca()
    test_pca_second_example()
    print("All tests passed")
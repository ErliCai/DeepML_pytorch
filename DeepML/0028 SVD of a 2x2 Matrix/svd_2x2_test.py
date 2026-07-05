import torch
from svd_2x2 import svd_2x2


def test_reconstruction():
    U, s, V = svd_2x2(torch.tensor([[-10.0, 8.0], [10.0, -1.0]]))
    result = U @ torch.diag(s) @ V
    assert torch.allclose(result, torch.tensor([[-10.0, 8.0], [10.0, -1.0]])), "Reconstruction failed"


def test_orthogonality():
    U, s, V = svd_2x2(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    print(U @ U.T,)
    assert torch.allclose(U @ U.T, torch.eye(2)), "U not orthogonal"


if __name__ == "__main__":
    test_reconstruction()
    test_orthogonality()
    print("All tests passed")

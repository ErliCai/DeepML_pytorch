import torch

from singular_value_decomposition_of_2x2_matrix import svd_2x2_singular_values


def test_svd_2x2_singular_values():
    A = torch.tensor([[2., 1.], [1., 2.]])

    U, S, Vt = svd_2x2_singular_values(A)

    assert torch.allclose(S, torch.tensor([3., 1.]), atol=1e-3)
    assert torch.allclose(U @ torch.diag(S) @ Vt, A, atol=1e-3)


def test_svd_2x2_singular_values_non_symmetric_matrix():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    U, S, Vt = svd_2x2_singular_values(A)

    expected_S = torch.linalg.svdvals(A)

    assert torch.allclose(S, expected_S, atol=1e-3)
    assert torch.allclose(U @ torch.diag(S) @ Vt, A, atol=1e-3)


if __name__ == "__main__":
    test_svd_2x2_singular_values()
    test_svd_2x2_singular_values_non_symmetric_matrix()
    print("All tests passed")
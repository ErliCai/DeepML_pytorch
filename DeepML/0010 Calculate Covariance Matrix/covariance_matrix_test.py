import torch

from covariance_matrix import calculate_covariance_matrix


def test_covariance_matrix():
    A = [[1, 2, 3], [4, 5, 6]]

    result = calculate_covariance_matrix(A)
    
    assert torch.allclose(result, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))


if __name__ == "__main__":
    test_covariance_matrix()
    print("All tests passed")

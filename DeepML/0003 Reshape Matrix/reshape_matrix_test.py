import torch

from reshape_matrix import reshape_matrix


def test_reshape_matrix():
    a = [[1,2,3,4],[5,6,7,8]]
    new_shape = (4, 2)

    result = reshape_matrix(a, new_shape)

    assert torch.equal(result, torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]))


if __name__ == "__main__":
    test_reshape_matrix()
    print("All tests passed")
import torch

from transpose_of_a_matrix import transpose_matrix


def test_transpose_matrix():
	a = [[1, 2, 3], [4, 5, 6]]

	result = transpose_matrix(a)

	assert torch.equal(result, torch.tensor([[1., 4.], [2., 5.], [3., 6.]]))


def test_transpose_square_matrix():
	a = [[1, 2], [3, 4]]

	result = transpose_matrix(a)

	assert torch.equal(result, torch.tensor([[1., 3.], [2., 4.]]))

if __name__ == "__main__":
	test_transpose_matrix()
	test_transpose_square_matrix()
	print("All tests passed")

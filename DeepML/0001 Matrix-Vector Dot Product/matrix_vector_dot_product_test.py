import torch
from matrix_vector_dot_product import matrix_dot_vector


def test_matrix_dot_vector():
    a = [[1, 2], [2, 4]]
    b = [1, 2]

    result = matrix_dot_vector(a, b)

    assert torch.equal(result, torch.tensor([5., 10.]))


def test_matrix_dot_vector_3_by_3():
    a = [[1, 2, 3], [2, 4, 5], [6, 8, 9]]
    b = [1, 2, 3]

    result = matrix_dot_vector(a, b)

    assert torch.equal(result, torch.tensor([14., 25., 49.]))


if __name__ == "__main__":

    test_matrix_dot_vector()
    test_matrix_dot_vector_3_by_3()
    print("All tests passed")
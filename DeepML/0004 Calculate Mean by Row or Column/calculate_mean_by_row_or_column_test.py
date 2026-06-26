import torch

from calculate_mean_by_row_or_column import calculate_matrix_mean


def test_calculate_mean_by_column():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mode = "column"

    result = calculate_matrix_mean(matrix, mode)

    assert torch.equal(result, torch.tensor([4.0, 5.0, 6.0]))


def test_calculate_mean_by_row():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mode = "row"

    result = calculate_matrix_mean(matrix, mode)

    assert torch.equal(result, torch.tensor([2.0, 5.0, 8.0]))


def test_calculate_mean_2_by_4_matrix():
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8]]

    column_result = calculate_matrix_mean(matrix, "column")
    row_result = calculate_matrix_mean(matrix, "row")

    assert torch.equal(column_result, torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(row_result, torch.tensor([2.5, 6.5]))


if __name__ == "__main__":
    test_calculate_mean_by_column()
    test_calculate_mean_by_row()
    test_calculate_mean_2_by_4_matrix()
    print("All tests passed")
import torch

from calculate_mean_by_row_or_column import calculate_mean


def test_calculate_mean_by_column():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mode = "column"

    result = calculate_mean(matrix, mode)

    assert torch.equal(result, torch.tensor([4.0, 5.0, 6.0]))


def test_calculate_mean_by_row():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mode = "row"

    result = calculate_mean(matrix, mode)

    assert torch.equal(result, torch.tensor([2.0, 5.0, 8.0]))


if __name__ == "__main__":
    test_calculate_mean_by_column()
    test_calculate_mean_by_row()
    print("All tests passed")
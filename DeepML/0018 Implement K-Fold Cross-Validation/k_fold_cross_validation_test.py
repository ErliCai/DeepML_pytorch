from k_fold_cross_validation import k_fold_cross_validation


def test_k_fold_cross_validation():
    result = k_fold_cross_validation(n_samples=10, k=5, shuffle=False)

    assert result == [
        ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
        ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]),
        ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]),
        ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
    ]


if __name__ == "__main__":
    test_k_fold_cross_validation()
    print("All tests passed")
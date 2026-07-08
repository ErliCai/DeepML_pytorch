import numpy as np

from adaboost_fit import adaboost_fit


def test_adaboost_fit():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)

    assert isinstance(clfs, list)
    assert len(clfs) == n_clf

    for clf in clfs:
        assert set(clf) == {"polarity", "threshold", "feature_index", "alpha"}
        assert clf["polarity"] in {-1, 1}
        assert isinstance(clf["feature_index"], int)
        assert isinstance(clf["alpha"], float)


if __name__ == "__main__":
    test_adaboost_fit()
    print("All tests passed")
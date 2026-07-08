import numpy as np
import math

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


def test_calculate_error():
    import torch
    from adaboost_fit import calculate_error
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = torch.tensor([1.0, 1.0, -1.0, -1.0])
    w = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # polarity=1, threshold=4, feature 0: preds[fv<4]=-1 → all wrong → error=1.0 → flipped to 0.0
    error, _ = calculate_error(w, X[:, 0], y, threshold=torch.tensor(4.0), polarity=1)

    print(error)
    assert abs(error - 0.0) < 1e-6

def test_calculate_alpha():
    import torch
    from adaboost_fit import calculate_alpha
    # error=0.25 → alpha = 0.5 * log(0.75/0.25) = 0.5 * log(3) ≈ 0.5493
    error = torch.tensor(0.25)
    alpha = calculate_alpha(error)
    assert abs(alpha - 0.5 * math.log(3)) < 1e-4


def test_platform_example():
    X = np.array([[8,7],[3,4],[5,9],[4,0],[1,0],[0,7],[3,8],[4,2],[6,8],[0,2]])
    y = np.array([1,-1,1,-1,1,-1,-1,-1,1,1])
    clfs = adaboost_fit(X, y, n_clf=2)
    expected = [
        {'polarity': 1,  'threshold': 4.0, 'feature_index': 0, 'alpha': 0.6931474},
        {'polarity': -1, 'threshold': 1.0, 'feature_index': 0, 'alpha': 0.7520389},
    ]
    print(clfs)
    for clf, exp in zip(clfs, expected):
        assert clf['polarity'] == exp['polarity']
        assert clf['feature_index'] == exp['feature_index']
        assert abs(clf['threshold'] - exp['threshold']) < 1e-4
        assert abs(clf['alpha'] - exp['alpha']) < 1e-4


if __name__ == "__main__":
    test_calculate_error()
    test_calculate_alpha()
    test_adaboost_fit()
    test_platform_example()
    print("All tests passed")
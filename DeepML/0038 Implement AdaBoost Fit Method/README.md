# [38. Implement AdaBoost Fit Method](https://www.deep-ml.com/problems/38)

## Problem Description

Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy array X of shape (n_samples, n_features) representing the dataset, a 1D numpy array y of shape (n_samples,) representing the labels, and an integer n_clf representing the number of classifiers. The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, and return a list of classifiers with their parameters.

## Example

Input:
```
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)
```
Output:
```
(example format, actual values may vary):
    # [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
    #  {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
    #  {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]
```

## Explanation

TODO

## Solution

TODO
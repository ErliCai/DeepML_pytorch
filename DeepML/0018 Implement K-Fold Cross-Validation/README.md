# [18. Implement K-Fold Cross-Validation](https://www.deep-ml.com/problems/18)

## Problem Description

Implement a function to generate train and test index splits for K-Fold Cross-Validation. Your task is to divide dataset indices into k folds and return a list of train-test index pairs for each fold.

Input:

n_samples: The total number of samples in the dataset (integer)
k: The number of folds (default 5)
shuffle: Whether to shuffle indices before splitting (default True)
Output:

A list of k tuples, where each tuple contains (train_indices, test_indices) as Python lists of integers
Requirements:

Split the indices into k roughly equal folds. If n_samples is not divisible by k, distribute the extra samples to the first folds (e.g., with 10 samples and k=3, fold sizes would be [4, 3, 3]).
For each fold iteration, use that fold as the test set and combine all other folds as the training set.
When shuffle=True, use np.random.shuffle() to randomize indices before splitting (the random seed will be set externally before calling your function).
When shuffle=False, keep indices in their original order [0, 1, 2, ..., n_samples-1].

## Example

Input:
```
k_fold_cross_validation(n_samples=10, k=5, shuffle=False)
```
Output:
```
[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
```

## Explanation

TODO

## Solution

TODO
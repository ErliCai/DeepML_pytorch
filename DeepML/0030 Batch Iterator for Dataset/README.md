# [30. Batch Iterator for Dataset](https://www.deep-ml.com/problems/30)

## Problem Description

Implement a batch iterable function that samples in a numpy array X and an optional numpy array y. The function should return batches of a specified size. If y is provided, the function should return batches of (X, y) pairs; otherwise, it should return batches of X only.

## Example

Input:
```
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1, 2, 3, 4, 5])
batch_size = 2
```
Output:
```
(array([[1, 2], [3, 4]]), array([1, 2]))
(array([[5, 6], [7, 8]]), array([3, 4]))
(array([[9, 10]]),        array([5]))
```

## Solution

TODO

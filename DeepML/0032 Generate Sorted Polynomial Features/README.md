# [32. Generate Sorted Polynomial Features](https://www.deep-ml.com/problems/32)

## Problem Description

Write a Python function that takes a 2-D NumPy array X and an integer degree, generates all polynomial feature combinations of the columns of X up to the given degree inclusive, then sorts the resulting features for each sample from lowest to highest value. The function should return a new 2-D NumPy array whose rows correspond to the input samples and whose columns are the ascending-sorted polynomial features.

## Example

Input:
```
X = np.array([[2, 3],
              [3, 4],
              [5, 6]])
degree = 2
output = polynomial_features(X, degree)
print(output)
```
Output:
```
[[ 1.  2.  3.  4.  6.  9.]
 [ 1.  3.  4.  9. 12. 16.]
 [ 1.  5.  6. 25. 30. 36.]]
```


## Solution

TODO

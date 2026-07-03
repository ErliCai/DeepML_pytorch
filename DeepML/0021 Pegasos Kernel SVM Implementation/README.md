# [21. Pegasos Kernel SVM Implementation](https://www.deep-ml.com/problems/21)

## Problem Description

Write a Python function that implements a deterministic version of the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. Note that while the original Pegasos algorithm is stochastic (it selects a single random sample at each step), this problem requires using all samples in every iteration (i.e., no random sampling). The function should perform binary classification and return the model's alpha coefficients as a list and bias as a float.

## Example

Input:
```
data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'linear', lambda_val = 0.01, iterations = 100
```
Output:
```
([2.0, 2.0, 6.0, 1.0], 36.2027)
```

## Explanation

TODO

## Solution

TODO: Briefly explain the implemented approach.

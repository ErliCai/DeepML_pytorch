# [10. Calculate Covariance Matrix](https://www.deep-ml.com/problems/10)

## Problem Description

Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists.

## Example

Input:
```
[[1, 2, 3], [4, 5, 6]]
```

Output:
```
[[1.0, 1.0], [1.0, 1.0]]
```

## Explanation

Given a data matrix $X$ of shape $(n\_samples, n\_features)$, the covariance matrix is calculated as:

**1. Mean-center the data:**

$$X_{centered} = X - \bar{X}$$

**2. Compute the covariance matrix:**

$$C = \frac{1}{n-1} X_{centered}^T X_{centered}$$

Where $C[i,i]$ is the variance of feature $i$ and $C[i,j]$ is the covariance between feature $i$ and feature $j$.

## Solution

Just follow above equation for implementation

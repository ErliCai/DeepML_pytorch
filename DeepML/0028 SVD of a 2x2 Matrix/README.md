# [28. SVD of a 2x2 Matrix](https://www.deep-ml.com/problems/28)

## Problem Description

Given a 2x2 matrix, compute its Singular Value Decomposition (SVD) without using built-in SVD functions.

Return matrices U, s, and V such that `A = U @ diag(s) @ V`.

- **U**: 2x2 orthogonal matrix (left singular vectors)
- **s**: 1D array of 2 singular values (non-negative, descending order)
- **V**: 2x2 matrix (right singular vectors, rows are eigenvectors of AᵀA)

## Example

Input:
```
A = [[-10, 8], [10, -1]]
```
Output:
```
U, s, V such that U @ diag(s) @ V ≈ A
```

## Explanation

1. Compute AᵀA (symmetric positive semidefinite)
2. Eigendecompose AᵀA → eigenvalues λ, eigenvectors V
3. Singular values: s = sqrt(λ), sorted descending
4. Left singular vectors: U = A @ V.T / s

## Solution

TODO

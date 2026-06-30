# [12. Singular Value Decomposition (SVD) of 2x2 Matrix](https://www.deep-ml.com/problems/12)

## Problem Description

Write a Python function that computes an approximate Singular Value Decomposition (SVD) of a real 2x2 matrix using one Jacobi rotation.

Input:

A: a NumPy array of shape (2, 2)
Rules:

You may use basic NumPy operations (matrix multiplication, transpose, element-wise math, etc.)
Do NOT call numpy.linalg.svd or any other high-level SVD routine
Use a single Jacobi rotation step (no iterative refinements)
Return: A tuple (U, S, Vt) where:

U is a 2x2 orthogonal matrix (left singular vectors)
S is a length-2 NumPy array containing the singular values
Vt is the transpose of the right singular vector matrix V
The decomposition should approximately satisfy: A = U @ diag(S) @ Vt

## Example

Input:
```
A = np.array([[2, 1], [1, 2]])
```
Output:
```
U ≈ [[0.707, -0.707], [0.707, 0.707]]
S = [3.0, 1.0]
Vt ≈ [[0.707, 0.707], [-0.707, 0.707]]
```

## Explanation

TODO: Add a high-level explanation of singular value decomposition.

## Solution

TODO: Briefly explain the implemented approach.
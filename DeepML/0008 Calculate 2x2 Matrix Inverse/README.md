# [8. Calculate 2x2 Matrix Inverse](https://www.deep-ml.com/problems/8)

## Problem Description

Write a Python function that calculates the inverse of a 2x2 matrix. The inverse of a matrix A is another matrix A_inv such that A * A_inv = I (the identity matrix).\n\nFor a 2x2 matrix [[a, b], [c, d]], the inverse exists only if the determinant (ad - bc) is non-zero.\n\nReturn None if the matrix is not invertible (i.e., when the determinant equals zero).

## Example

Input:
matrix = [[4, 7], [2, 6]]
Output:
[[0.6, -0.7], [-0.2, 0.4]]

## Explanation

$A^{-1}$ is the matrix that undoes $A$ — multiplying them together gives the identity matrix: $A \cdot A^{-1} = I$.

When $ A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ and determinant of A is non zero

We can calculate it with:

$A^{-1} = \frac{1}{\text{Det(A)}}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

We can quickly verify that this is indeed the inverse of A by multiplying them together

$A A^{-1} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{ad-bc}\begin{bmatrix} ad-bc & 0 \\ 0 & ad-bc \end{bmatrix} = I$

## Solution

We first ensure the determinant is non-zero and then use the above formula to calculate the inverse of the matrix.
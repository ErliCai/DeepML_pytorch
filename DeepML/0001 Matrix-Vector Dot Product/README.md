# [1. Matrix-Vector Dot Product](https://www.deep-ml.com/problems/1)

## Problem Description
Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. For example, an n x m matrix requires a vector of length m.

You may assume the input matrix is a valid (non-jagged) list of lists and the vector is a non-empty list.

## Example

Input:
```
a = [[1, 2], [2, 4]], b = [1, 2]
```
Output:
```
[5, 10]
```
Reasoning:
Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5; Row 2: (2 * 1) + (4 * 2) = 2 + 8 = 10

## Explanation

We are essentially implementing matrix multiplication for the special case where the first input is a 2-D matrix and the second input is a 1-D vector.

PyTorch has built-in functions like `matmul` and `mv` that satisfy this requirement. You can also use the `@` operator.

```python
a = torch.tensor([[1, 2], [2, 4]])
b = torch.tensor([1, 2])

torch.matmul(a, b)
torch.mv(a, b)
a @ b
```

## Solution

We loop over the first dimension of the matrix, treating each row as one vector. For each row, we multiply its values with the matching values in the input vector, sum those products, and store the result in the output vector.


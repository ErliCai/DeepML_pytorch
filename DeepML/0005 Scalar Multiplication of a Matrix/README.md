# [5. Scalar Multiplication of a Matrix](https://www.deep-ml.com/problems/5)

## Problem Description

Write a Python function that multiplies a matrix by a scalar and returns the result.

## Example

Input:
matrix = [[1, 2], [3, 4]], scalar = 2
Output:
[[2, 4], [6, 8]]

## Explanation

Scalar multiplication means multiplying every element in the matrix by the same scalar value. In PyTorch, this can be done directly because tensor operations are applied element by element.

```python
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
scalar = 2

matrix * scalar  # tensor([[2., 4.], [6., 8.]])
```

## Solution

Too easy I don't even want to explain

We first convert the input matrix into a PyTorch tensor and create an empty result tensor with the same shape. Then we loop through each row and column, multiply each value by the scalar, and store it in the matching position of the result.
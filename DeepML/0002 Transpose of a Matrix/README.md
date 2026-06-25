# [2. Transpose of a Matrix](https://www.deep-ml.com/problems/2)

## Problem Description

Write a Python function that computes the transpose of a given 2D matrix. The transpose of a matrix is formed by turning its rows into columns and columns into rows. For an m×n matrix, the transpose will be an n×m matrix.

## Example

Input:
a = [[1, 2, 3], [4, 5, 6]]
Output:
[[1, 4], [2, 5], [3, 6]]

Reasoning:
The input is a 2×3 matrix. The transpose swaps rows and columns: the first row [1, 2, 3] becomes the first column, and the second row [4, 5, 6] becomes the second column, resulting in a 3×2 matrix

## Explanation

For a 2-D matrix, transpose means swapping rows and columns.

PyTorch has a general function, `torch.transpose`, that can swap any two axes of a tensor. For a simple 2-D matrix, we can also use the tensor property `a.T`.

```python
a = torch.tensor([[1, 2], [3, 4]])

torch.transpose(a, 0, 1)
a.T
```


## Solution

We create a new empty matrix with the row and column sizes swapped. Then we loop through each value in the original matrix and place it at the swapped position in the result, so `a[i, j]` becomes `result[j, i]`.
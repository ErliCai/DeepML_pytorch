# [4. Calculate Mean by Row or Column](https://www.deep-ml.com/problems/4)

## Problem Description

Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

## Example

Input:
```
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
```

Output:
```
[4.0, 5.0, 6.0]
```

## Explanation

Here, we are basically implementing a special case of PyTorch's `mean` function, which can calculate means along any axis or multiple axes. We can think of it as collapsing on the dimensions by averaging its values.

For a 2-D matrix, `dim=0` calculates the mean of each column, while `dim=1` calculates the mean of each row.

```python
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

torch.mean(matrix, dim=0)  # tensor([4., 5., 6.])
torch.mean(matrix, dim=1)  # tensor([2., 5., 8.])
```

For a 3-D tensor, the same idea applies: `dim` tells PyTorch which axis to average over.

```python
x = torch.tensor([
	[[1., 2., 3.], [4., 5., 6.]],
	[[7., 8., 9.], [10., 11., 12.]]
])

torch.mean(x, dim=0)       # shape: [2, 3]
torch.mean(x, dim=1)       # shape: [2, 3]
torch.mean(x, dim=2)       # shape: [2, 2]
torch.mean(x, dim=(1, 2))  # tensor([3.5000, 9.5000])
```

## Solution

We first convert the input matrix into a PyTorch tensor. If the mode is `column`, we loop through each column, sum its values, and divide by the number of rows. If the mode is `row`, we loop through each row, sum its values, and divide by the number of columns.
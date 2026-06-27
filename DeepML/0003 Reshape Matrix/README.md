# [3. Reshape Matrix](https://www.deep-ml.com/problems/3)

## Problem Description

Write a Python function that reshapes a given matrix into a specified shape. if it can't be reshaped return back an empty list []

## Example

Input:
```
a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
```

Output:
```
[[1, 2], [3, 4], [5, 6], [7, 8]]
```

## Explanation

Reshaping works by reading all elements in row-major order (left to right, top to bottom), then filling them into the new shape the same way. Here we are essentially implementing a special case of PyTorch's reshape function that can reshape tensors into arbitrary shape as long as the total number of elements matches. Below is an example usage of the PyTorch reshape function.

```python
a = torch.tensor([[1,2,3,4],[5,6,7,8]])
new_shape = (4, 2)

a.reshape(new_shape)
```

## Solution


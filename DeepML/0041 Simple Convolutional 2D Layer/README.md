# [41. Simple Convolutional 2D Layer](https://www.deep-ml.com/problems/41)

## Problem Description

In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.

## Example

Input:
```
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
```
Output:
```
[[ 1.  1. -4.],[ 9.  7. -4.],[ 0. 14. 16.]]
```

## Explanation

TODO

## Solution

TODO

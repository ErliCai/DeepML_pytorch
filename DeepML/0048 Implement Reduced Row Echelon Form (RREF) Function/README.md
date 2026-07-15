# [48. Implement Reduced Row Echelon Form (RREF) Function](https://www.deep-ml.com/problems/48)

## Problem Description

In this problem, your task is to implement a function that converts a given matrix into its Reduced Row Echelon Form (RREF). The RREF of a matrix is a special form where each leading entry in a row is 1, and all other elements in the column containing the leading 1 are zeros, except for the leading 1 itself.

However, there are some additional details to keep in mind:

- Diagonal entries can be 0 if the matrix is reducible (i.e., the row corresponding to that position can be eliminated entirely).
- Some rows may consist entirely of zeros.
- If a column contains a pivot (a leading 1), all other entries in that column should be zero.

Your task is to implement the RREF algorithm, which must handle these cases and convert any given matrix into its RREF.

## Example

Input:
```
import numpy as np

matrix = np.array([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])

rref_matrix = rref(matrix)
print(rref_matrix)
```
Output:
```
# array([
#    [ 1.  0.  0. -8.],
#    [ 0.  1.  0.  1.],
#    [-0. -0.  1. -2.]
# ])
```

## Explanation

TODO

## Solution

TODO

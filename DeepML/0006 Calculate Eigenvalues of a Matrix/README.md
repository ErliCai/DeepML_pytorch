# [6. Calculate Eigenvalues of a Matrix](https://www.deep-ml.com/problems/6)

## Problem Description

Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

## Example

Input:
matrix = [[2, 1], [1, 2]]
Output:
[3.0, 1.0]

Reasoning:
The eigenvalues of the matrix are calculated using the characteristic equation. For a 2x2 matrix, the equation is:

$$
\lambda^2 - \text{trace}(A)\lambda + \det(A) = 0
$$

where $\lambda$ represents the eigenvalues.

## Explanation

We are implementing a function to calculate eigenvalues for a 2x2 matrix.

If you are not familiar with linear algebra concepts such as trace, determinant, eigenvalues, or eigenvectors, this video gives a good visual explanation: [“Eigenvectors and eigenvalues”](https://www.3blue1brown.com/lessons/eigenvalues) from 3Blue1Brown.

PyTorch also provides a built-in function for calculating eigenvalues:

```python
matrix = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)

eigen = torch.linalg.eigvals(matrix)
```

## Solution

As explained in the reference above, eigenvalues of a 2x2 matrix can be calculated from the characteristic equation:

$$
\lambda^2 - \text{trace}(A)\lambda + \det(A) = 0
$$

We can then solve this quadratic equation using the **quadratic formula**:

$$
\lambda = \frac{\text{trace}(A) \pm \sqrt{\text{trace}(A)^2 - 4\det(A)}}{2}
$$
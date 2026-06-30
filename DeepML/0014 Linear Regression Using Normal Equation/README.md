# [14. Linear Regression Using Normal Equation](https://www.deep-ml.com/problems/14)

## Problem Description

Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

## Example

Input:
```
X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
```
Output:
```
[0.0, 1.0]
```

## Explanation

The Normal Equation is the closed-form solution to linear regression — instead of iteratively nudging weights toward the minimum (gradient descent), it solves for the optimal weights directly using linear algebra.

Linear regression minimizes the MSE loss:

$$L(w) = \frac{1}{n}\|y - Xw\|^2$$

This is smooth and convex, so the minimum occurs where the gradient is zero. Taking the derivative with respect to $w$ and setting it to zero gives:

$$X^T X w = X^T y$$

Solving for $w$:

$$w = (X^T X)^{-1} X^T y$$

We can't simply solve $w = X^{-1}y$ because $X$ is rarely square — with more samples (rows) than features (columns), the system is overdetermined, so $X^{-1}$ doesn't exist. $X^T X$, however, is always square ($n\_features \times n\_features$) and invertible as long as the features are linearly independent, so it gives the least-squares best fit even when no exact solution exists.

In the example, $X$'s first column of 1s is the bias term and the second column is the feature $x = [1,2,3]$. Since $y = [1,2,3]$ matches $x$ exactly, the perfect fit is $y = x$, giving $w_0 = 0$ and $w_1 = 1$.

## Solution

TODO: Briefly explain the implemented approach.

# [15. Linear Regression Using Gradient Descent](https://www.deep-ml.com/problems/15)

## Problem Description

Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays `X` and `y` as input, along with the learning rate `alpha` and the number of `iterations`. Return the learned coefficients, or weights, as a NumPy array.

The input matrix `X` has shape `(m, n)`, where `m` is the number of training examples and `n` is the number of features, including the bias column of ones for the intercept. The target vector `y` has shape `(m,)`.

Requirements:

- Initialize all weights to zero.
- Use batch gradient descent, meaning every update uses all training examples.
- Minimize the Mean Squared Error (MSE) loss function:

$$
L(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$

where:

$$
h_\theta(x) = X\theta
$$

Here, `m` is the number of samples. The factor of $\frac{1}{2}$ simplifies the gradient calculation.

## Example

Input:
```
X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([3, 5, 7]), alpha = 0.1, iterations = 1000
```
Output:
```
[1.0, 2.0]
```

Reasoning:
The data follows y = 1 + 2x perfectly. Starting from theta = [0, 0], gradient descent iteratively updates the weights until converging to approximately [1, 2], representing an intercept of 1 and slope of 2.

## Explanation

TODO: Add a high-level explanation of batch gradient descent for linear regression.

## Solution

TODO: Briefly explain the implemented approach.
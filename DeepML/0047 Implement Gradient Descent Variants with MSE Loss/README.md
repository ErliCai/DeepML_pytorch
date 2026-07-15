# [47. Implement Gradient Descent Variants with MSE Loss](https://www.deep-ml.com/problems/47)

## Problem Description

In this problem, you need to implement a single function that can perform three variants of gradient descent: Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use.

### Requirements

- Do not shuffle the data; process samples in their original order (index 0, 1, 2, ...)
- For **Batch GD**: use all samples to compute a single gradient update per epoch
- For **Stochastic GD**: iterate through each sample sequentially (i.e., process sample 0, then 1, then 2, etc.) — not randomly selected
- For **Mini-Batch GD**: form batches from consecutive samples without overlap (e.g., for batch_size=2: first batch uses indices [0,1], second batch uses [2,3], etc.)
- The `n_epochs` parameter specifies how many complete passes through the dataset to perform
- For each epoch, process all samples according to the specified method

## Example

Input:
```
import numpy as np

# Sample data
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# Parameters
learning_rate = 0.01
n_epochs = 1000
batch_size = 2

# Initialize weights
weights = np.zeros(X.shape[1])

# Test Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='batch')
# Test Stochastic Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, method='stochastic')
# Test Mini-Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size, method='mini_batch')
```
Output:
```
[float, float]
[float, float]
[float, float]
```

## Explanation

TODO

## Solution

TODO

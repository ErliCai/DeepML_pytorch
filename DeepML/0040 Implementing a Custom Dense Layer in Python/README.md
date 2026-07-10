# [40. Implementing a Custom Dense Layer in Python](https://www.deep-ml.com/problems/40)

## Problem Description

You are provided with a base `Layer` class that defines the structure of a neural network layer. Your task is to implement a subclass called `Dense`, which represents a fully connected neural network layer. The `Dense` class should extend the `Layer` class and implement the following methods:

1. **Initialization (`__init__`)**:
   - Define the layer with a specified number of neurons (`n_units`) and an optional input shape (`input_shape`).
   - Set up placeholders for the layer's weights (`W`), biases (`w0`), and optimizers.
2. **Weight Initialization (`initialize`)**:
   - Initialize the weights `W` using a uniform distribution with a limit of `1 / sqrt(input_shape[0])`, and bias `w0` should be set to zero.
   - Initialize optimizers for `W` and `w0`.
3. **Parameter Count (`parameters`)**:
   - Return the total number of trainable parameters in the layer, which includes the parameters in `W` and `w0`.
4. **Forward Pass (`forward_pass`)**:
   - Compute the output of the layer by performing a dot product between the input `X` and the weight matrix `W`, and then adding the bias `w0`.
5. **Backward Pass (`backward_pass`)**:
   - Calculate and return the gradient with respect to the input.
   - If the layer is trainable, update the weights and biases using the optimizer's update rule.
6. **Output Shape (`output_shape`)**:
   - Return the shape of the output produced by the forward pass, which should be `(self.n_units,)`.

**Objective**:
Extend the `Layer` class by implementing the `Dense` class to ensure it functions correctly within a neural network framework.

## Example

Input:
```
# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)
```
Output:
```
Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
Backward pass output: [[ 0.00129588  0.00953634]]
```

## Explanation

TODO

## Solution

TODO

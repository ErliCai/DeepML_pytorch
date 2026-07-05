# [26. Implementing Basic Autograd Operations](https://www.deep-ml.com/problems/26)

## Problem Description

Inspired by Andrej Karpathy's micrograd check out his excellent video: https://youtu.be/VMj-3S1tku0

Implement a Value class that wraps a scalar number and supports automatic differentiation. Your class should:

Support + and * operations that work with other Value objects or plain numbers, returning a new Value with the correct result.

Support a relu() method that applies the ReLU activation function (outputs the value if positive, 0 otherwise).

Support a backward() method that computes gradients for all upstream Value objects in the computation graph using backpropagation.

Each operation should track its inputs and know how to locally compute its gradient contribution. The backward() method should process nodes in the correct order so that gradients accumulate properly from output back to inputs.

Hints:

Each operation creates a new Value that remembers which values produced it (_children).
Think about what order nodes need to be processed during the backward pass.
Gradients should be accumulated (+=), not replaced.

## Example

Input:
```
a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
```
Output:
```
Value(data=2, grad=0) Value(data=-3, grad=0) Value(data=10, grad=0)
```

## Explanation

TODO

## Solution

TODO

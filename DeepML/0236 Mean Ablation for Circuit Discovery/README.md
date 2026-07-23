# [236. Mean Ablation for Circuit Discovery](https://www.deep-ml.com/problems/236)

## Problem Description

Implement a mean ablation function for neural network interpretability. Mean ablation is a technique used to isolate neural circuits by replacing a node's activation with its mean value computed over a reference distribution. Given node activations, a binary mask indicating which nodes to ablate, and precomputed mean activations, return the ablated activations where masked nodes are replaced with their means while unmasked nodes retain their original values.

## Example

Input:

```
activations = [0.5, -0.3, 0.8, 0.2], mask = [1, 0, 1, 0], means = [0.1, 0.0, 0.2, -0.1]
```

Output:

```
[0.1, -0.3, 0.2, 0.2]
```

## Explanation

Nodes at indices 0 and 2 have `mask=1`, so they are ablated and replaced with means 0.1 and 0.2. Nodes at indices 1 and 3 have `mask=0`, so they keep their original values -0.3 and 0.2.

## Solution

TODO

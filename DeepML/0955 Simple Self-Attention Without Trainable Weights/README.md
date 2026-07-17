# [955. Simple Self-Attention Without Trainable Weights](https://www.deep-ml.com/problems/955)

## Problem Description

Implement a simplified self-attention mechanism that computes context vectors for each token in an input sequence, without any trainable weights.

Given an input matrix X of shape (T, d) where each row is a d-dimensional token embedding, compute context vectors Z of shape (T, d) as follows:

For each query token i, compute attention scores by taking the dot product of x_i with every token x_j in the sequence (including itself). This yields a (T, T) score matrix.
Normalize each row of the score matrix using softmax to obtain attention weights that sum to 1.
Each context vector z_i is the weighted sum of all input tokens x_j, using the attention weights from row i.
Return the resulting context matrix as a list of lists, with each value rounded in the test (not in your solution).

## Example

Input:
X = [[1.0, 0.0], [0.0, 1.0]]
Output:
[[0.7311, 0.2689], [0.2689, 0.7311]]

## Explanation

TODO

## Solution

TODO

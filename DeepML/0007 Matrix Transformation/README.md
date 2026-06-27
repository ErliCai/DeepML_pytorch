# [7. Matrix Transformation](https://www.deep-ml.com/problems/7)

## Problem Description

Write a Python function that transforms a given matrix A using the operation $T^{-1}AS$, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1

## Example

Input:
A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
Output:
[[0.5,1.5],[1.5,3.5]]

## Explanation

This is called change of basis in linear algebra, check out this video [
Change of basis | Chapter 13, Essence of linear algebra](https://www.youtube.com/watch?v=P2LTAUO1TdA) from 3Blue1Brown for the mathematical implication of change of basis 

## Solution

First check if T and S are invertible by computing their determinants — if either is zero, return -1. Otherwise, compute T⁻¹ using `torch.linalg.inv` and return the matrix product `T⁻¹ @ A @ S`.

# [52. Implement Recall Metric in Binary Classification](https://www.deep-ml.com/problems/52)

## Problem Description

Task: Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function recall(y_true, y_pred) that calculates the recall metric. The function should accept two inputs:

y_true: A list of true binary labels (0 or 1) for the dataset.
y_pred: A list of predicted binary labels (0 or 1) from the model.
Your function should return the recall value as a float. If the denominator (TP + FN) is zero, return 0.0 to avoid division by zero.

## Example

Input:
```
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))
```
Output:
```
0.75
```


## Explanation

TODO

## Solution

TODO

# [20. Decision Tree Learning](https://www.deep-ml.com/problems/20)

## Problem Description

Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.

Tie-Breaking Rules:

If multiple attributes have equal information gain, choose the one that appears first in the attributes list.
If a leaf node has equal counts of different classes, return the class that comes first alphabetically.
Process attribute values in sorted order to ensure consistent tree structure.

## Example

Input:
```
examples = [
    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
],
attributes = ['Outlook', 'Wind'],
target_attr = 'PlayTennis'
```
Output:
```
{'Outlook': {'Overcast': 'Yes', 'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}}, 'Sunny': 'No'}}
```

## Explanation

TODO

## Solution

TODO
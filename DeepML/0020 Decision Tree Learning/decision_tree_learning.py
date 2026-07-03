import torch
import math
from collections import Counter
from typing import List, Dict, Any, Union


def calculate_entropy(labels: List[Any]) -> float:
    """
    Compute the Shannon entropy of the list of labels.
    labels: list of any hashable items.
    Returns a Python float.
    """

    count = {}
    for label in labels:
        if label not in count:
            count[label] = 1
        else:
            count[label] += 1
    total_count = len(labels)
    for label in count:
        count[label] /= total_count
    
    entropy = sum([(-prob) * math.log2(prob) for label, prob in count.items()])
    return entropy

def calculate_information_gain(
    examples: List[Dict[str, Any]],
    attr: str,
    target_attr: str
) -> float:
    """
    Compute information gain for splitting `examples` on `attr` w.r.t. `target_attr`.
    Returns a Python float.
    """

    example_labels = [example[target_attr] for example in examples]
    entropy = calculate_entropy(example_labels)
    split = set([example[attr] for example in examples])

    entropy_after_split = 0
    for s in split:
        labels = [example[target_attr] for example in examples if example[attr] == s]
        entropy_S = calculate_entropy(labels)
        entropy_after_split += len(labels) * entropy_S / len(example_labels)

    return entropy - entropy_after_split

def majority_class(
    examples: List[Dict[str, Any]],
    target_attr: str
) -> Any:
    """
    Return the most common value of `target_attr` in `examples`.
    In case of a tie, return the class that comes first alphabetically.
    """

    value_count = {}
    for example in examples:
        val = example[target_attr]
        if val in value_count:
            value_count[val] += 1
        else:
            value_count[val] = 1
    
    sorted_val = sorted(value_count.items(), key=lambda item: (-item[1], item[0]))
    return sorted_val[0][0]


def learn_decision_tree(
    examples: List[Dict[str, Any]],
    attributes: List[str],
    target_attr: str
) -> Union[Dict[str, Any], Any]:
    """
    Learn a decision tree using the ID3 algorithm.
    Returns either a nested dict representing the tree or a class label at the leaves.
    """

    labels = set([example[target_attr] for example in examples])
    if len(labels) == 1:
        return labels.pop()

    if len(attributes) == 0:
        return majority_class(examples, target_attr)

    highest_attr = None
    highest_gain = -1
    for attribute in attributes:
        info_gain = calculate_information_gain(examples, attribute, target_attr)
        if info_gain > highest_gain:
            highest_attr = attribute
            highest_gain = info_gain

    tree = {highest_attr: {}}
    ## split dataset
    attrs_set = set([example[highest_attr] for example in examples])
    remaining_attr = [attr for attr in attributes if attr != highest_attr]
    for attr in attrs_set:
        example_split = [example for example in examples if example[highest_attr] == attr]
        tree[highest_attr][attr] = learn_decision_tree(example_split, remaining_attr, target_attr)

    return tree

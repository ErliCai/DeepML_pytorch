import torch
import math
from typing import List, Dict

def adaboost_fit(X, y, n_clf) -> List[Dict]:
    """
    Fit an AdaBoost classifier using PyTorch tensors (no sklearn).
    Args:
        X: torch.Tensor or array-like, shape (n_samples, n_features)
        y: torch.Tensor or array-like, shape (n_samples,) with labels (+1/-1)
        n_clf: int, number of weak classifiers
    Returns:
        List of dictionaries with classifier params: 'polarity', 'threshold', 'feature_index', 'alpha'.
    """
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    # initialize classifier
    weight = torch.tensor([1/len(X) for i in range(len(X))])
    classifier = [{'polarity':1, 'threshold':0, 'feature_index':0, 'alpha':0} for i in range(n_clf)]

    for n in range(n_clf):
        lowest = (0,0)
        h_x = None
        lowest_error = 1
        lowest_popularity = 1
        for feature_index in range(len(X[0])):
            for threshold in X_t[:,feature_index]:
                
                error, prediction, popularity= calculate_error(weight, X_t, y_t, threshold, feature_index)

                if error < lowest_error:
                    lowest_error = error
                    lowest = (threshold, feature_index)
                    h_x = prediction
                    lowest_popularity = popularity

        alpha = calculate_alpha(lowest_error)
        if lowest_popularity == -1:
            h_x = -h_x
        weight = update_weight(weight, alpha, y_t, h_x)
        classifier[n]["polarity"] = lowest_popularity
        classifier[n]["threshold"] = float(lowest[0].item())
        classifier[n]["feature_index"] = lowest[1]
        classifier[n]["alpha"] = alpha.item()

    return classifier


def calculate_error(weight, X, y, threshold, idx):

    predictions = X[:,idx] > threshold
    predictions = [1 if prediction else -1 for prediction in predictions]
    predictions = torch.tensor(predictions)
    misclassified = (y != predictions)
    error = weight[misclassified].sum()

    popularity = 1
    if error > 0.5:
        error = 1 - error
        popularity = -1

    return error, predictions, popularity

def calculate_alpha(error):

    return 1/2 * torch.log((1-error)/ (error + 1e-10))

def update_weight(w, alpha, y, h):

    w = w * torch.exp(-alpha * y * h)
    w = w/w.sum()

    return w

## official implementation

import torch
import math
from typing import List, Dict

def adaboost_fit2(X, y, n_clf) -> List[Dict]:
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    n_samples, n_features = X.shape
    w = torch.full((n_samples,), 1.0 / n_samples)
    clfs = []
    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')
        # Find best decision stump
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = torch.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds = torch.ones(n_samples)
                    preds[polarity * feature_values < polarity * threshold] = -1
                    error = torch.sum(w * (preds != y))
                    if error > 0.5:
                        error = 1 - error
                        polarity = -polarity
                    if error < min_error:
                        min_error = error.item()
                        clf['polarity'] = polarity
                        clf['threshold'] = threshold.item()
                        clf['feature_index'] = feature_i
        # Compute alpha
        clf['alpha'] = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        # Update weights
        feature_values = X[:, clf['feature_index']]
        preds = torch.ones(n_samples)
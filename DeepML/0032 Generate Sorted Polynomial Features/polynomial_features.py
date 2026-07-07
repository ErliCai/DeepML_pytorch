import torch
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial feature combinations
    (with constant term), sorted for each sample from smallest to largest.
    """
    
    rst = []
    for x in X:
        i = 0
        poly = torch.tensor([1.0])

        last_poly = torch.tensor([1.0])
        while i < degree:
            new_poly = []
            for feature in x:
                new_poly.append(last_poly * feature)

            new_poly = torch.cat(new_poly, dim = 0)
            new_poly = torch.unique(new_poly, dim=0)
            last_poly = new_poly
            poly = torch.cat([poly,new_poly], dim=0)
            i += 1
        rst.append(poly)


    rst = torch.stack(rst)
    return rst

def polynomial_features2(X, degree):
    X = torch.as_tensor(X, dtype=torch.float32)
    n_samples, n_features = X.shape
    combs = list(combinations_with_replacement(range(n_features), degree))
    # For all powers from 0 up to degree
    all_combs = [(c,) for c in range(n_features)]
    for d in range(2, degree+1):
        all_combs.extend(list(combinations_with_replacement(range(n_features), d)))
    # Always include bias (all 0 degree)
    features = [torch.ones(n_samples, 1)]
    for comb in all_combs:
        vals = torch.prod(X[:, list(comb)], dim=1, keepdim=True)
        features.append(vals)
    result = torch.cat(features, dim=1)
    # Sort each row
    result_sorted, _ = torch.sort(result, dim=1)
    return result_sorted
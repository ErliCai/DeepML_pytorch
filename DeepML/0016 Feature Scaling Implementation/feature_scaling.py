import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)

    ## standardization

    data_standardized = torch.empty_like(data_t)

    for i in range(len(data_t[0])):
        col = data_t[:,i]
        mean, std = torch.mean(col), torch.std(col, unbiased=False)

        print(mean, std)

        data_standardized[:,i] = (col- mean) /std

    ## minmax

    data_minmax = torch.empty_like(data_t)
    for i in range(len(data_t[0])):
        col = data_t[:,i]
        x_min, x_max = col.min(), col.max()

        data_minmax[:,i] = (col- x_min) / (x_max - x_min)
    
    return data_standardized.round(decimals=4), data_minmax.round(decimals=4)


# mean = torch.mean(data_t, dim=0)
# std = torch.std(data_t, dim=0, unbiased=False)

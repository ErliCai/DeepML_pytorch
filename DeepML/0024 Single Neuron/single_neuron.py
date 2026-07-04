import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE

    features_t = torch.tensor(features, dtype=torch.float)
    labels_t = torch.tensor(labels, dtype=torch.float)
    weights_t = torch.tensor(weights, dtype=torch.float)
    bias_t = torch.tensor(bias, dtype=torch.float)

    z = torch.matmul(features_t, weights_t) + bias_t
    sigmoid = torch.sigmoid(z)
    mse = torch.nn.functional.mse_loss(sigmoid, labels_t)

    sigmoid, mse = sigmoid.tolist(), mse.item()

    sigmoid = [round(i, 4) for i in sigmoid]
    mse = round(mse, 4)
    # print(sigmoid, mse)

    return sigmoid, mse
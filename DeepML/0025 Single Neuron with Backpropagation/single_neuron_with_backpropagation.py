import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    weights = initial_weights.clone().requires_grad_(True)
    bias = torch.tensor(initial_bias, requires_grad=True)

    mse = []

    for _ in range(epochs):

        z = features @ weights + bias
        sigmoids = torch.sigmoid(z)

        loss = nn.functional.mse_loss(sigmoids, labels)
        mse.append(round(loss.item(),4))
        loss.backward()
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

    updated_weights = [round(weight, 4) for weight in weights.tolist()]
    updated_bias = round(bias.item(),4)

    # print(updated_weights, updated_bias, mse)

    return updated_weights, updated_bias, mse

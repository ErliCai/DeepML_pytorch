import torch
import torch.nn as nn
import torch.nn.functional as F

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


def train_neuron2(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
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

    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    initial_weights = torch.as_tensor(initial_weights, dtype=torch.float32)   

    neuron = nn.Linear(features.shape[1], 1, bias=True)

    with torch.no_grad():
        neuron.weight.copy_(initial_weights.unsqueeze(0))
        neuron.bias.fill_(initial_bias)

    optimizer = torch.optim.SGD(neuron.parameters(), lr = learning_rate)

    mse_value = []

    for _ in range(epochs):
        optimizer.zero_grad()

        predictions = torch.sigmoid(neuron(features).squeeze())

        loss = F.mse_loss(predictions, labels)
        mse_value.append(round(loss.item(), 4))

        loss.backward()
        optimizer.step()

    updated_weights = [round(w, 4) for w in neuron.weight.data.squeeze().tolist()]
    updated_bias = round(neuron.bias.data.item(), 4)

    return updated_weights, updated_bias, mse_value

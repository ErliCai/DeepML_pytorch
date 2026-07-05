import math
import torch

from single_neuron_with_backpropagation import train_neuron


def test_single_neuron_with_backpropagation():
    features = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = torch.tensor([1, 0, 0], dtype=torch.float32)
    initial_weights = torch.tensor([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2

    updated_weights, updated_bias, mse_values = train_neuron(
        features,
        labels,
        initial_weights,
        initial_bias,
        learning_rate,
        epochs,
    )

    assert updated_weights == [0.1036, -0.1425]
    assert math.isclose(updated_bias, -0.0167, abs_tol=1e-4)
    assert mse_values == [0.3033, 0.2942]


if __name__ == "__main__":
    test_single_neuron_with_backpropagation()
    print("All tests passed")
import numpy as np
import torch

from dense_layer import Dense


class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad


def test_dense_layer_backward():
    layer = Dense(n_units=3, input_shape=(2,))
    layer.initialize(MockOptimizer())

    X = np.array([[1, 2]])
    _ = layer.forward_pass(X)

    accum = np.array([[0.1, 0.2, 0.3]])
    back_out = layer.backward_pass(accum)
    result = back_out.detach().numpy().round(8)
    print(result)
    assert np.allclose(result, np.array([[0.20816524, -0.22928937]]), atol=1e-4)


if __name__ == "__main__":
    test_dense_layer_backward()
    print("All tests passed")

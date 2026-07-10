import numpy as np

from dense_layer import Dense


class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad


def test_dense_layer():
    dense_layer = Dense(n_units=3, input_shape=(2,))

    optimizer = MockOptimizer()
    dense_layer.initialize(optimizer)

    X = np.array([[1, 2]])
    output = dense_layer.forward_pass(X)
    print("Forward pass output:", output)
    assert np.allclose(
        output, np.array([[-0.00655782, 0.01429615, 0.00905812]]), atol=1e-4
    )

    accum_grad = np.array([[0.1, 0.2, 0.3]])
    back_output = dense_layer.backward_pass(accum_grad)
    print("Backward pass output:", back_output)
    assert np.allclose(
        back_output, np.array([[0.00129588, 0.00953634]]), atol=1e-4
    )


if __name__ == "__main__":
    test_dense_layer()
    print("All tests passed")

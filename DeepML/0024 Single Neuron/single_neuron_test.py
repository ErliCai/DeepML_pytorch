from single_neuron import single_neuron_model


def test_single_neuron():
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0, 1, 0]
    weights = [0.7, -0.4]
    bias = -0.1

    result = single_neuron_model(features, labels, weights, bias)

    assert result == ([0.4626, 0.4134, 0.6682], 0.3349)


if __name__ == "__main__":
    test_single_neuron()
    print("All tests passed")
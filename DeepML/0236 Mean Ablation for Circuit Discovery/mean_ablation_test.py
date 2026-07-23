import torch

from mean_ablation import mean_ablate


def test_mean_ablate_example():
    activations = torch.tensor([0.5, -0.3, 0.8, 0.2])
    mask = torch.tensor([1, 0, 1, 0])
    means = torch.tensor([0.1, 0.0, 0.2, -0.1])
    result = mean_ablate(activations, mask, means)
    expected = torch.tensor([0.1, -0.3, 0.2, 0.2])
    print(result)
    assert torch.allclose(result, expected)


def test_mean_ablate_batch():
    activations = torch.tensor(
        [[0.5, -0.3, 0.8, 0.2], [0.4, 0.6, -0.2, 0.9]]
    )
    mask = torch.tensor([1, 0, 1, 0])
    means = torch.tensor([0.1, 0.0, 0.2, -0.1])
    result = mean_ablate(activations, mask, means)
    expected = torch.tensor(
        [[0.1, -0.3, 0.2, 0.2], [0.1, 0.6, 0.2, 0.9]]
    )
    print(result)
    assert torch.allclose(result, expected)


def test_mean_ablate_no_nodes():
    activations = torch.tensor([0.5, -0.3, 0.8, 0.2])
    mask = torch.zeros(4, dtype=torch.int64)
    means = torch.tensor([0.1, 0.0, 0.2, -0.1])
    result = mean_ablate(activations, mask, means)
    print(result)
    assert torch.equal(result, activations)


if __name__ == "__main__":
    test_mean_ablate_example()
    test_mean_ablate_batch()
    test_mean_ablate_no_nodes()
    print("All tests passed")

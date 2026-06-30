import torch

from feature_scaling import feature_scaling


def test_feature_scaling():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    standardized_data, normalized_data = feature_scaling(data)

    expected_standardized = torch.tensor([
        [-1.2247, -1.2247],
        [0.0, 0.0],
        [1.2247, 1.2247],
    ])
    expected_normalized = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    assert torch.allclose(standardized_data, expected_standardized, atol=1e-4)
    assert torch.allclose(normalized_data, expected_normalized, atol=1e-4)


if __name__ == "__main__":
    test_feature_scaling()
    print("All tests passed")
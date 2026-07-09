import torch

from ridge_loss import ridge_loss

def test_ridge_loss():
    X = torch.tensor([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=torch.float)
    w = torch.tensor([0.2, 2], dtype=torch.float)
    y_true = torch.tensor([2, 3, 4, 5], dtype=torch.float)
    alpha = 0.1

    loss = ridge_loss(X, w, y_true, alpha)

    assert torch.isclose(loss, torch.tensor(2.204))


if __name__ == "__main__":
    test_ridge_loss()
    print("All tests passed")
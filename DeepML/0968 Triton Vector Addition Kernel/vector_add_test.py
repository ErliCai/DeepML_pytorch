import torch

from vector_add import vector_add


def test_vector_add_small():
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = torch.tensor([10.0, 20.0, 30.0], device="cuda")
    result = vector_add(x, y)
    print(result)
    assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0], device="cuda"))


if __name__ == "__main__":
    test_vector_add_small()
    print("All tests passed")

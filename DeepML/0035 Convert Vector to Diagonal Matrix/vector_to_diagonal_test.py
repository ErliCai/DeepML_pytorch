import numpy as np
import torch

from vector_to_diagonal import make_diagonal


def test_convert_vector_to_diagonal():
    x = np.array([1, 2, 3])
    output = make_diagonal(x)

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ],
        dtype=torch.float,
    )

    assert torch.equal(output, expected)


if __name__ == "__main__":
    test_convert_vector_to_diagonal()
    print("All tests passed")
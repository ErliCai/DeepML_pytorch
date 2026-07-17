import numpy as np

from self_attention import self_attention


def test_self_attention_basic():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = self_attention(X)
    print(result)
    expected = np.array([[0.7311, 0.2689], [0.2689, 0.7311]])
    assert np.allclose(result, expected, atol=1e-4)


def test_self_attention_single_row():
    X = [[1.0, 1.0]]
    result = self_attention(X)
    print([[round(v, 4) for v in row] for row in result])
    expected = np.array([[1.0, 1.0]])
    assert np.allclose(result, expected, atol=1e-4)


if __name__ == "__main__":
    test_self_attention_basic()
    test_self_attention_single_row()
    print("All tests passed")

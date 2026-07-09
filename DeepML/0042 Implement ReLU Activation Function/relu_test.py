from relu import relu


def test_relu():
    assert relu(0) == 0.0
    assert relu(1) == 1.0
    assert relu(-1) == 0.0


if __name__ == "__main__":
    test_relu()
    print("All tests passed")
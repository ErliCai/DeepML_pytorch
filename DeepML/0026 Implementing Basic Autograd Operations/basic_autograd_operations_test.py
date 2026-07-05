from basic_autograd_operations import Value


def test_basic_autograd_operations():
    a = Value(2)
    b = Value(-3)
    c = Value(10)
    d = a + b * c
    e = d.relu()
    e.backward()

    assert repr(a) == "Value(data=2, grad=0)"
    assert repr(b) == "Value(data=-3, grad=0)"
    assert repr(c) == "Value(data=10, grad=0)"


if __name__ == "__main__":
    test_basic_autograd_operations()
    print("All tests passed")

from basic_autograd_operations import Value


def test_basic_autograd_operations():
    a = Value(2); b = Value(3); c = Value(10)
    d = a + b * c
    e = Value(7) * Value(2)
    f = e + d
    g = f.relu()
    g.backward()

    assert repr(a) == "Value(data=2, grad=1)"
    assert repr(b) == "Value(data=3, grad=10)"
    assert repr(c) == "Value(data=10, grad=3)"
    assert repr(d) == "Value(data=32, grad=1)"
    assert repr(e) == "Value(data=14, grad=1)"
    assert repr(f) == "Value(data=46, grad=1)"
    assert repr(g) == "Value(data=46, grad=1)"


if __name__ == "__main__":
    test_basic_autograd_operations()
    print("All tests passed")

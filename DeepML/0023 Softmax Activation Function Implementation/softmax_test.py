from softmax import softmax


def test_softmax():
    scores = [1, 2, 3]

    result = softmax(scores)

    assert [round(value, 4) for value in result] == [0.0900, 0.2447, 0.6652]


if __name__ == "__main__":
    test_softmax()
    print("All tests passed")
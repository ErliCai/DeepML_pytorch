from transformation_matrix import transform_basis


def test_transform_basis():
    B = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    C = [[1, 2.3, 3],
         [4.4, 25, 6],
         [7.4, 8, 9]]

    result = transform_basis(B, C)

    expected = [[-0.6772, -0.0126,  0.2342],
                [-0.0184,  0.0505, -0.0275],
                [ 0.5732, -0.0345, -0.0569]]

    assert result == expected


if __name__ == "__main__":
    test_transform_basis()
    print("All tests passed")

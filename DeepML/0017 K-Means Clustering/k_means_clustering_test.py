from k_means_clustering import k_means_clustering


def test_k_means_clustering():
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10

    result = k_means_clustering(points, k, initial_centroids, max_iterations)

    assert result == [(1, 2), (10, 2)]


if __name__ == "__main__":
    test_k_means_clustering()
    print("All tests passed")
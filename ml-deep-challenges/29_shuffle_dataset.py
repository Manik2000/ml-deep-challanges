import numpy as np


def shuffle_data(X, y, seed=None):
    """Shuffle data using Fisher-Yates method"""
    if seed:
        np.random.seed(seed)
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    for i in range(n_samples - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return X[idx], y[idx]


def test_shuffle_data(expected, X, y, seed=None):
    X_shuffled, y_shuffled = shuffle_data(X, y, seed)
    assert np.allclose(X_shuffled, expected[0])
    assert np.allclose(y_shuffled, expected[1])


if __name__ == "__main__":
    test_shuffle_data(
        (np.array([[3, 4], [7, 8], [1, 2], [5, 6]]), np.array([2, 4, 1, 3])),
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        np.array([1, 2, 3, 4]),
        seed=42,
    )
    test_shuffle_data(
        (np.array([[4, 4], [2, 2], [1, 1], [3, 3]]), np.array([40, 20, 10, 30])),
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        np.array([10, 20, 30, 40]),
        seed=24,
    )

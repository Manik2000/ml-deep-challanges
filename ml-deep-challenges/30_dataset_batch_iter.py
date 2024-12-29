import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    batches = []
    for i in range(0, n_samples, batch_size):
        if y is not None:
            batches.append([X[i : i + batch_size], y[i : i + batch_size]])
        else:
            batches.append(X[i : i + batch_size])
    return batches


def test_batch_iterator(truth_value, X, y=None, batch_size=64):
    vals = batch_iterator(X, y, batch_size)
    if y is not None:
        for (true_x, true_y), (x, y) in zip(truth_value, vals):
            assert np.allclose(true_x, x)
            assert np.allclose(true_y, y)
    else:
        for true_x, x in zip(truth_value, vals):
            assert np.allclose(true_x, x)


if __name__ == "__main__":
    test_batch_iterator(
        [
            (np.array([[1, 1], [2, 2], [3, 3]]), np.array([1, 2, 3])),
            (np.array([[4, 4]]), np.array([4])),
        ],
        np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        np.array([1, 2, 3, 4]),
        batch_size=3,
    )

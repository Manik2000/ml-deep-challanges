import numpy as np


def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    np.random.shuffle(data)
    folds = np.array_split(data, k)
    k_folds = []
    for i in range(k):
        train = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        test = folds[i]
        k_folds.append([train, test])
    return k_folds


def test_cross_validation_split(expected, data, k, seed=42):
    k_folds = cross_validation_split(data, k, seed)
    for (train_true, test_true), (train, test) in zip(expected, k_folds):
        assert np.allclose(train_true, train)
        assert np.allclose(test_true, test)


if __name__ == "__main__":
    test_cross_validation_split(
        [
            [[[9, 10], [5, 6], [1, 2], [7, 8]], [[3, 4]]],
            [[[3, 4], [5, 6], [1, 2], [7, 8]], [[9, 10]]],
            [[[3, 4], [9, 10], [1, 2], [7, 8]], [[5, 6]]],
            [[[3, 4], [9, 10], [5, 6], [7, 8]], [[1, 2]]],
            [[[3, 4], [9, 10], [5, 6], [1, 2]], [[7, 8]]],
        ],
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        5,
        42,
    )
    test_cross_validation_split(
        [
            [[[1, 2], [7, 8]], [[3, 4], [9, 10], [5, 6]]],
            [[[3, 4], [9, 10], [5, 6]], [[1, 2], [7, 8]]],
        ],
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        2,
        42,
    )

import numpy as np


def to_categorical(x: np.ndarray, n_col: int = None):
    if not n_col:
        n_col = len(np.unique(x))
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


if __name__ == "__main__":
    assert np.array_equal(
        to_categorical(np.array([0, 1, 2, 1, 0])),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    )

import numpy as np
from utils import compare_two_arrays


def divide_on_feature(
    X: np.ndarray, feature_i: int, threshold: int | float
) -> list[np.ndarray]:
    mask = X[:, feature_i] >= threshold
    return [X[mask, :], X[~mask, :]]


if __name__ == "__main__":
    result = divide_on_feature(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 0, 5
    )
    expected = [np.array([[5, 6], [7, 8], [9, 10]]), np.array([[1, 2], [3, 4]])]
    assert compare_two_arrays(result[0], expected[0])
    assert compare_two_arrays(result[1], expected[1])

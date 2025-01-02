import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y_true)
    SST = np.sum((y_true - y_mean) ** 2)
    SSR = np.sum((y_true - y_pred) ** 2)
    return 1 - SSR / SST


if __name__ == "__main__":
    assert (
        r_squared(np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.1, 2.9, 4.2, 4.8]))
        == 0.989
    )

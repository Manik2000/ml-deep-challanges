import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    assert accuracy_score(np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0])) == 0.5
    assert (
        accuracy_score(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1]))
        == 0.8333333333333334
    )

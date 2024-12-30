import numpy as np


def precision(y_true: np.array, y_pred: np.array):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    return tp / (tp + fp) if tp + fp > 0 else 0


if __name__ == "__main__":
    assert precision(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1])) == 1.0
    assert precision(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 1])) == 0.5

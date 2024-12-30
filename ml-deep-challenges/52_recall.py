import numpy as np


def recall(y_true: np.array, y_pred: np.array):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return round(tp / (tp + fn), 3) if tp + fn > 0 else 0


if __name__ == "__main__":
    assert recall(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 1, 0, 0, 1])) == 0.75
    assert recall(np.array([1, 0, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 0, 1])) == 0.333

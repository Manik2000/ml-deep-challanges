import numpy as np


def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    if precision + recall == 0:
        return 0
    return round(
        (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall), 3
    )


if __name__ == "__main__":
    assert (
        f_score(
            y_true=np.array([1, 0, 1, 1, 0, 1]),
            y_pred=np.array([1, 0, 1, 0, 0, 1]),
            beta=1,
        )
        == 0.857
    )

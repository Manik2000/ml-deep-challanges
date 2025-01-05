import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    preds = X @ w
    return np.mean((preds - y_true) ** 2) + alpha * w @ w.T


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    w = np.array([0.2, 2])
    y_true = np.array([2, 3, 4, 5])
    alpha = 0.1

    loss = ridge_loss(X, w, y_true, alpha)
    assert np.isclose(loss, 2.204, atol=1e-3)

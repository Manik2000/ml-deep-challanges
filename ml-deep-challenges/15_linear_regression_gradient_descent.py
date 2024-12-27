import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    for _ in range(iterations):
        y_hat = X @ theta
        error = y_hat - y
        gradient = X.T @ error / m
        theta -= alpha * gradient
    return np.round(theta, 4).flatten()


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            linear_regression_gradient_descent(
                np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000
            ),
            np.array([0.1107, 0.9513]),
        )
    )

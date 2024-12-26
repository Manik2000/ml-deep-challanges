import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    X, y = np.array(X), np.array(y)
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    theta = np.round(theta, 4)
    theta[np.abs(theta) <= 1e-6] = -0
    return theta.tolist()


if __name__ == "__main__":
    assert linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]) == [
        -0.0,
        1.0,
    ]
    assert linear_regression_normal_equation(
        [[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]
    ) == [4.0, -1.0, -0.0]

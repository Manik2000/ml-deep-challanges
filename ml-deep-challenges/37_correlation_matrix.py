import numpy as np


def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray = None):
    if Y is None:
        Y = X
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    n = X.shape[0]
    covariance_matrix = X_centered.T @ Y_centered / n
    D_x = np.diag(1 / np.std(X, axis=0))
    D_y = np.diag(1 / np.std(Y, axis=0))
    return D_x @ covariance_matrix @ D_y


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        )
    )
    assert np.all(
        np.isclose(
            calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])),
            np.array(
                [
                    [1.0, 0.84298868, 0.8660254],
                    [0.84298868, 1.0, 0.46108397],
                    [0.8660254, 0.46108397, 1.0],
                ]
            ),
        )
    )

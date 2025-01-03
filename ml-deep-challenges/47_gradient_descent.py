import numpy as np
from utils import compare_two_arrays


def gradient_descent(
    X, y, weights, learning_rate, n_iterations, batch_size=1, method="batch"
):
    n_samples, n_features = X.shape
    final_weights = weights.copy()
    for _ in range(n_iterations):
        if method == "batch":
            y_pred = np.dot(X, final_weights)
            error = y_pred - y
            final_weights -= learning_rate * np.dot(X.T, error) / n_samples * 2
        elif method == "stochastic":
            for i in range(n_samples):
                X_i = X[i, :].reshape(1, n_features)
                y_i = y[i]
                y_pred = np.dot(X_i, final_weights)
                error = y_pred - y_i
                final_weights -= learning_rate * np.dot(X_i.T, error) * 2
        elif method == "mini_batch":
            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                y_pred = np.dot(X_batch, final_weights)
                error = y_pred - y_batch
                final_weights -= (
                    learning_rate * np.dot(X_batch.T, error) / batch_size * 2
                )
    return final_weights


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])

    learning_rate = 0.01
    n_iterations = 1000
    batch_size = 2
    weights = np.zeros(X.shape[1])

    assert compare_two_arrays(
        gradient_descent(X, y, weights, learning_rate, n_iterations, method="batch"),
        np.array([1.01003164, 0.97050576]),
    )

    assert compare_two_arrays(
        gradient_descent(
            X, y, weights, learning_rate, n_iterations, method="stochastic"
        ),
        np.array([1.00000058, 0.99999813]).reshape(1, -1),
    )

    assert compare_two_arrays(
        gradient_descent(
            X, y, weights, learning_rate, n_iterations, batch_size, method="mini_batch"
        ),
        np.array([1.0003804, 0.99883421]).reshape(1, -1),
    )

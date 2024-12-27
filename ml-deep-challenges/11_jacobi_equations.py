import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    length = A.shape[0]
    diag = np.diag(A)
    x = np.zeros(length)
    for _ in range(n):
        x = (b - np.dot(A, x) + diag * x) / diag
    return np.round(x, 4).tolist()


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            solve_jacobi(
                np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]), 2
            ),
            [0.146, 0.2032, -0.5175],
        )
    )
    assert np.all(
        np.isclose(
            solve_jacobi(
                np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]), 5
            ),
            [-0.0806, 0.9324, 2.4422],
        )
    )

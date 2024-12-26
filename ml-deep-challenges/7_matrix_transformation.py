import numpy as np


def transform_matrix(
    A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]
) -> list[list[int | float]]:
    A, S, T = (
        np.array(A, dtype=np.float64),
        np.array(S, dtype=np.float64),
        np.array(T, dtype=np.float64),
    )
    epsilon = np.finfo(np.float64).eps
    if np.linalg.cond(S) > 1 / epsilon or np.linalg.cond(T) > 1 / epsilon:
        return -1
    return np.linalg.solve(T, A @ S).tolist()


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]),
            [[0.5, 1.5], [1.5, 3.5]],
        )
    )
    assert np.all(
        np.isclose(
            transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]),
            [[-4.0, 2.0], [3.0, -1.0]],
        )
    )

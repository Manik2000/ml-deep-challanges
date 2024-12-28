import numpy as np
from utils import time_func


def make_diagonal(x: np.ndarray) -> np.ndarray:
    return np.diag(x)


def make_diagonal_v2(x: np.ndarray) -> np.ndarray:
    identity = np.eye(x.shape[0])
    return identity * x


@time_func
def make_diagonal_check(func: callable, x: np.ndarray) -> np.ndarray:
    return func(x)


if __name__ == "__main__":
    assert np.all(
        make_diagonal(np.array([1, 2, 3]))
        == np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    )
    assert np.all(
        make_diagonal(np.array([4, 5, 6, 7]))
        == np.array([[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]])
    )

    benchmark_vector = np.random.randint(0, 100, 10_000)
    make_diagonal_check(make_diagonal, benchmark_vector)
    make_diagonal_check(make_diagonal_v2, benchmark_vector)

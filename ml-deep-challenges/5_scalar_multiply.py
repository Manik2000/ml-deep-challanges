import random

from utils import time_func


def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    n, m = len(matrix), len(matrix[0])
    return [[matrix[i][j] * scalar for j in range(m)] for i in range(n)]


def scalar_multiply_2(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    return [[element * scalar for element in row] for row in matrix]


@time_func
def scalar_multiply_check(
    func: callable, matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    return func(matrix, scalar)


if __name__ == "__main__":
    assert scalar_multiply([[1, 2], [3, 4]], 2) == [[2, 4], [6, 8]]
    assert scalar_multiply([[0, -1], [1, 0]], -1) == [[0, 1], [-1, 0]]

    benchmark_matrix = [
        [random.randint(0, 100) for _ in range(10_000)] for _ in range(10_000)
    ]
    scalar_multiply_check(scalar_multiply, benchmark_matrix, 2)
    scalar_multiply_check(scalar_multiply_2, benchmark_matrix, 2)

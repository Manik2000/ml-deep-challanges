import random

from utils import time_func


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    n, m = len(matrix), len(matrix[0])
    if mode == "row":
        return [sum(row) / m for row in matrix]
    return [sum(col) / n for col in zip(*matrix)]


def calculate_matrix_mean_2(matrix: list[list[float]], mode: str) -> list[float]:
    n, m = len(matrix), len(matrix[0])
    if mode == "row":
        return [sum(row) / m for row in matrix]
    return [sum(matrix[i][j] for i in range(n)) / n for j in range(n)]


@time_func
def calculate_matrix_mean_check(
    func: callable, matrix: list[list[float]], mode: str
) -> list[float]:
    return func(matrix, mode)


if __name__ == "__main__":
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column") == [
        4.0,
        5.0,
        6.0,
    ]
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "row") == [
        2.0,
        5.0,
        8.0,
    ]

    benchmark_matrix = [
        [random.randint(0, 100) for _ in range(10_000)] for _ in range(10_000)
    ]
    calculate_matrix_mean_check(calculate_matrix_mean, benchmark_matrix, "row")
    calculate_matrix_mean_check(calculate_matrix_mean_2, benchmark_matrix, "row")

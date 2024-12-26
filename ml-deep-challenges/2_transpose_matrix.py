import random

from utils import time_func


def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    n, m = len(a), len(a[0])
    transposed = [[None for _ in range(n)] for _ in range(m)]
    for i in range(n):
        for j in range(m):
            transposed[j][i] = a[i][j]
    return transposed


def transpose_matrix_2(a: list[list[int | float]]) -> list[list[int | float]]:
    return [list(i) for i in zip(*a)]


@time_func
def transpose_check(
    func: callable, a: list[list[int | float]]
) -> list[list[int | float]]:
    return func(a)


if __name__ == "__main__":
    assert transpose_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

    benchmark_matrix = [
        [random.randint(0, 100) for _ in range(1000)] for _ in range(1000)
    ]

    transpose_check(transpose_matrix, benchmark_matrix)
    transpose_check(transpose_matrix_2, benchmark_matrix)
